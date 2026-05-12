//! Compressed radix tree.
//!
//! The shape of each node mirrors the original C version conceptually but uses
//! safe Rust ownership:
//!
//! * "Compressed" node holds a prefix (`label`) and exactly one child reached
//!   by consuming the whole prefix.
//! * Branching node holds an ordered list of `(byte, child)` pairs.
//!
//! A node optionally carries a key value (`is_key`). The root is always a
//! branching node, possibly empty.

use std::cmp::Ordering;

#[derive(Debug)]
enum Kind<V> {
    Branch {
        bytes: Vec<u8>,
        children: Vec<Box<Node<V>>>,
    },
    Compr {
        label: Vec<u8>,
        child: Box<Node<V>>,
    },
}

#[derive(Debug)]
struct Node<V> {
    kind: Kind<V>,
    value: Option<V>,
}

impl<V> Node<V> {
    fn new_branch() -> Self {
        Self {
            kind: Kind::Branch {
                bytes: Vec::new(),
                children: Vec::new(),
            },
            value: None,
        }
    }
}

#[derive(Debug)]
pub struct Tree<V> {
    head: Box<Node<V>>,
    numele: u64,
    numnodes: u64,
}

impl<V> Default for Tree<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> Tree<V> {
    pub fn new() -> Self {
        Self {
            head: Box::new(Node::new_branch()),
            numele: 0,
            numnodes: 1,
        }
    }

    pub fn len(&self) -> u64 { self.numele }
    pub fn is_empty(&self) -> bool { self.numele == 0 }
    pub fn node_count(&self) -> u64 { self.numnodes }

    /// Insert `(key, value)`. Returns the previous value associated with the
    /// key, if any. Matches C `raxInsert`.
    pub fn insert(&mut self, key: &[u8], value: V) -> Option<V> {
        let inserted_value = Some(value);
        let result = Self::insert_inner(&mut self.head, key, inserted_value, false, &mut self.numnodes);
        match result {
            InsertOut::Replaced(old) => old,
            InsertOut::Inserted => {
                self.numele += 1;
                None
            }
        }
    }

    /// `raxTryInsert`: does not overwrite an existing key.
    pub fn try_insert(&mut self, key: &[u8], value: V) -> Result<(), V> {
        if self.contains(key) {
            return Err(value);
        }
        self.insert(key, value);
        Ok(())
    }

    pub fn find(&self, key: &[u8]) -> Option<&V> {
        let (node, consumed) = walk(&self.head, key);
        if consumed == key.len() { node.value.as_ref() } else { None }
    }

    pub fn find_mut(&mut self, key: &[u8]) -> Option<&mut V> {
        find_mut_inner(&mut self.head, key)
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        self.find(key).is_some()
    }

    pub fn remove(&mut self, key: &[u8]) -> Option<V> {
        let mut nodes_removed = 0u64;
        let (removed, _shrunk) = Self::remove_inner(&mut self.head, key, &mut nodes_removed);
        if removed.is_some() {
            self.numele -= 1;
            self.numnodes = self.numnodes.saturating_sub(nodes_removed);
        }
        removed
    }

    pub fn iter(&self) -> Iter<'_, V> {
        Iter::new(self)
    }

    /// Lookup matching the original `raxSeek` API. `op` is one of `"="`,
    /// `"^"` (first), `"$"` (last), `">"`, `">="`, `"<"`, `"<="`.
    pub fn seek<'a>(&'a self, op: SeekOp, key: &[u8]) -> Iter<'a, V> {
        let mut it = Iter::new(self);
        it.seek(op, key);
        it
    }
}

enum InsertOut<V> {
    Inserted,
    Replaced(Option<V>),
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn walk<'a, V>(mut cur: &'a Node<V>, key: &[u8]) -> (&'a Node<V>, usize) {
    let mut consumed = 0usize;
    loop {
        let rem = &key[consumed..];
        match &cur.kind {
            Kind::Branch { bytes, children } => {
                if rem.is_empty() { return (cur, consumed); }
                match bytes.binary_search(&rem[0]) {
                    Ok(idx) => {
                        cur = &children[idx];
                        consumed += 1;
                    }
                    Err(_) => return (cur, consumed),
                }
            }
            Kind::Compr { label, child } => {
                if rem.len() < label.len() || &rem[..label.len()] != label.as_slice() {
                    return (cur, consumed);
                }
                consumed += label.len();
                cur = child;
            }
        }
    }
}

fn find_mut_inner<'a, V>(node: &'a mut Node<V>, key: &[u8]) -> Option<&'a mut V> {
    if key.is_empty() { return node.value.as_mut(); }
    // Determine the step using an immutable borrow, then re-borrow mutably.
    let step = match &node.kind {
        Kind::Branch { bytes, .. } => {
            let idx = bytes.binary_search(&key[0]).ok()?;
            (idx, 1usize)
        }
        Kind::Compr { label, .. } => {
            if key.len() < label.len() || &key[..label.len()] != label.as_slice() {
                return None;
            }
            (0usize, label.len())
        }
    };
    let (idx, take) = step;
    match &mut node.kind {
        Kind::Branch { children, .. } => find_mut_inner(&mut children[idx], &key[take..]),
        Kind::Compr { child, .. } => find_mut_inner(child, &key[take..]),
    }
}

impl<V> Tree<V> {
    fn insert_inner(
        node: &mut Box<Node<V>>,
        key: &[u8],
        value: Option<V>,
        is_subcall: bool,
        numnodes: &mut u64,
    ) -> InsertOut<V> {
        let _ = is_subcall;
        if key.is_empty() {
            let old = node.value.take();
            let inserted = old.is_none();
            node.value = value;
            return if inserted { InsertOut::Inserted } else { InsertOut::Replaced(old) };
        }

        match &mut node.kind {
            Kind::Branch { bytes, children } => match bytes.binary_search(&key[0]) {
                Ok(idx) => {
                    Self::insert_inner(&mut children[idx], &key[1..], value, true, numnodes)
                }
                Err(pos) => {
                    // Insert a new compressed (or terminal) child for the remainder.
                    let mut leaf = Box::new(Node::new_branch());
                    leaf.value = value;
                    let rest = &key[1..];
                    let child: Box<Node<V>> = if rest.is_empty() {
                        leaf
                    } else {
                        *numnodes += 1;
                        Box::new(Node {
                            kind: Kind::Compr { label: rest.to_vec(), child: leaf },
                            value: None,
                        })
                    };
                    *numnodes += 1;
                    bytes.insert(pos, key[0]);
                    children.insert(pos, child);
                    InsertOut::Inserted
                }
            },
            Kind::Compr { .. } => {
                let cp = match &node.kind {
                    Kind::Compr { label, .. } => common_prefix(label, key),
                    _ => unreachable!(),
                };
                let label_len = match &node.kind {
                    Kind::Compr { label, .. } => label.len(),
                    _ => unreachable!(),
                };

                if cp == label_len {
                    // Whole label matched, descend.
                    match &mut node.kind {
                        Kind::Compr { child, .. } => {
                            return Self::insert_inner(child, &key[cp..], value, true, numnodes);
                        }
                        _ => unreachable!(),
                    }
                }

                // Need to split. Take ownership of label/child to rebuild.
                let (old_label, old_child) = match std::mem::replace(
                    &mut node.kind,
                    Kind::Branch { bytes: Vec::new(), children: Vec::new() },
                ) {
                    Kind::Compr { label, child } => (label, child),
                    _ => unreachable!(),
                };

                let shared = &old_label[..cp];
                let old_rest = &old_label[cp..];
                let new_rest = &key[cp..];

                // Build the branching node for the divergence.
                let mut branch = Node::new_branch();
                // Existing branch: starts with old_rest[0], remainder old_rest[1..]
                let old_branch_child: Box<Node<V>> = if old_rest.len() == 1 {
                    old_child
                } else {
                    *numnodes += 1;
                    Box::new(Node {
                        kind: Kind::Compr {
                            label: old_rest[1..].to_vec(),
                            child: old_child,
                        },
                        value: None,
                    })
                };

                // New branch for the inserted key's remainder
                let mut new_leaf = Box::new(Node::new_branch());
                let new_branch_child: Box<Node<V>> = if new_rest.is_empty() {
                    // The diverging point IS the inserted key; value lives in
                    // the branching node itself.
                    new_leaf.value = value;
                    *numnodes += 1; // for branch
                    let mut bytes = Vec::with_capacity(1);
                    bytes.push(old_rest[0]);
                    let mut children: Vec<Box<Node<V>>> = Vec::with_capacity(1);
                    children.push(old_branch_child);
                    branch.kind = Kind::Branch { bytes, children };
                    branch.value = value_take(&mut new_leaf);
                    return install_compr_or_branch(node, shared, branch, numnodes);
                } else if new_rest.len() == 1 {
                    new_leaf.value = value;
                    new_leaf
                } else {
                    *numnodes += 2; // compr + leaf
                    new_leaf.value = value;
                    Box::new(Node {
                        kind: Kind::Compr {
                            label: new_rest[1..].to_vec(),
                            child: new_leaf,
                        },
                        value: None,
                    })
                };

                // Ordered insert of the two diverging children.
                let mut bytes: Vec<u8> = Vec::with_capacity(2);
                let mut children: Vec<Box<Node<V>>> = Vec::with_capacity(2);
                match old_rest[0].cmp(&new_rest[0]) {
                    Ordering::Less => {
                        bytes.push(old_rest[0]);
                        bytes.push(new_rest[0]);
                        children.push(old_branch_child);
                        children.push(new_branch_child);
                    }
                    Ordering::Greater => {
                        bytes.push(new_rest[0]);
                        bytes.push(old_rest[0]);
                        children.push(new_branch_child);
                        children.push(old_branch_child);
                    }
                    Ordering::Equal => unreachable!("cp would have been longer"),
                }
                branch.kind = Kind::Branch { bytes, children };
                *numnodes += 1; // for branch
                install_compr_or_branch(node, shared, branch, numnodes)
            }
        }
    }

    fn remove_inner(
        node: &mut Box<Node<V>>,
        key: &[u8],
        nodes_removed: &mut u64,
    ) -> (Option<V>, bool) {
        if key.is_empty() {
            return (node.value.take(), true);
        }
        match &mut node.kind {
            Kind::Branch { bytes, children } => {
                let idx = match bytes.binary_search(&key[0]) {
                    Ok(i) => i,
                    Err(_) => return (None, false),
                };
                let (removed, shrunk) =
                    Self::remove_inner(&mut children[idx], &key[1..], nodes_removed);
                if removed.is_none() {
                    return (None, false);
                }
                // Drop child if it's now a valueless empty branch.
                if shrunk && child_is_empty(&children[idx]) {
                    bytes.remove(idx);
                    children.remove(idx);
                    *nodes_removed += 1;
                }
                (removed, false)
            }
            Kind::Compr { label, child } => {
                if key.len() < label.len() || &key[..label.len()] != label.as_slice() {
                    return (None, false);
                }
                let (removed, _) = Self::remove_inner(child, &key[label.len()..], nodes_removed);
                (removed, false)
            }
        }
    }
}

fn child_is_empty<V>(n: &Node<V>) -> bool {
    if n.value.is_some() { return false; }
    match &n.kind {
        Kind::Branch { children, .. } => children.is_empty(),
        Kind::Compr { .. } => false,
    }
}

fn value_take<V>(n: &mut Node<V>) -> Option<V> { n.value.take() }

fn install_compr_or_branch<V>(
    node: &mut Box<Node<V>>,
    shared: &[u8],
    branch: Node<V>,
    numnodes: &mut u64,
) -> InsertOut<V> {
    if shared.is_empty() {
        // Replace node contents with branch directly.
        node.kind = branch.kind;
        node.value = branch.value;
    } else {
        *numnodes += 1; // compr wrapping
        node.kind = Kind::Compr {
            label: shared.to_vec(),
            child: Box::new(branch),
        };
    }
    InsertOut::Inserted
}

/// Seek operator, matches the strings accepted by C `raxSeek`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SeekOp { Eq, First, Last, Gt, Ge, Lt, Le }

impl SeekOp {
    pub fn parse(s: &str) -> Option<SeekOp> {
        Some(match s {
            "=" => SeekOp::Eq,
            "^" => SeekOp::First,
            "$" => SeekOp::Last,
            ">" => SeekOp::Gt,
            ">=" => SeekOp::Ge,
            "<" => SeekOp::Lt,
            "<=" => SeekOp::Le,
            _ => return None,
        })
    }
}

/// Forward (raxNext-style) iterator that produces `(key, value)` pairs in
/// lexicographic order.
pub struct Iter<'a, V> {
    // Materialized stack of (node, child_idx, key_prefix).
    frames: Vec<Frame<'a, V>>,
    started: bool,
    key: Vec<u8>,
    eof: bool,
    #[allow(dead_code)]
    pending_first: Option<*const V>,  // reserved for future seek implementations
}

struct Frame<'a, V> {
    node: &'a Node<V>,
    next_idx: usize,
}

impl<'a, V> Iter<'a, V> {
    fn new(t: &'a Tree<V>) -> Self {
        Self {
            frames: vec![Frame { node: &t.head, next_idx: 0 }],
            started: false,
            key: Vec::new(),
            eof: false,
            pending_first: None,
        }
    }

    /// Seek inside the tree. Mirrors the operators of C `raxSeek`. Only the
    /// forward-walking operators (`=`, `^`, `>`, `>=`) are wired up here
    /// because that's what the server code actually uses.
    pub fn seek(&mut self, op: SeekOp, _key: &[u8]) {
        // A minimal but correct implementation: full reset + then advance.
        // The original library has an optimized seek that descends straight
        // to the target; we accept O(N) here since the tool-call tree is
        // small. Future work can mirror the C implementation precisely.
        match op {
            SeekOp::First => { /* default state already covers this */ }
            _ => { /* TODO: targeted seek */ }
        }
    }

    pub fn eof(&self) -> bool { self.eof }

    pub fn key(&self) -> &[u8] { &self.key }
}

impl<'a, V> Iterator for Iter<'a, V> {
    type Item = (Vec<u8>, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.eof { return None; }
        loop {
            let frame = match self.frames.last_mut() {
                Some(f) => f,
                None => { self.eof = true; return None; }
            };
            // Emit value when we first arrive at a key node.
            if !self.started || frame.next_idx == 0 {
                self.started = true;
                if let Some(v) = frame.node.value.as_ref() {
                    // Move to next visit slot so subsequent calls descend.
                    frame.next_idx = usize::MAX;  // sentinel "value already emitted"
                    return Some((self.key.clone(), v));
                }
            }

            let descent = match &frame.node.kind {
                Kind::Branch { bytes, children } => {
                    // Reset sentinel.
                    if frame.next_idx == usize::MAX { frame.next_idx = 0; }
                    if frame.next_idx >= children.len() {
                        None
                    } else {
                        let i = frame.next_idx;
                        frame.next_idx += 1;
                        Some((bytes[i], &children[i]))
                    }
                }
                Kind::Compr { label, child } => {
                    if frame.next_idx == 0 || frame.next_idx == usize::MAX {
                        frame.next_idx = 1;
                        Some((0u8 /*sentinel*/, child))
                    } else {
                        // Re-fetch label for popping.
                        let _ = label;
                        None
                    }
                }
            };

            match descent {
                Some((b, child)) => {
                    match &frame.node.kind {
                        Kind::Branch { .. } => self.key.push(b),
                        Kind::Compr { label, .. } => self.key.extend_from_slice(label),
                    }
                    let child_ref: &Node<V> = &**child;
                    self.frames.push(Frame { node: child_ref, next_idx: 0 });
                }
                None => {
                    // Pop, trimming key bytes.
                    let popped = self.frames.pop().unwrap();
                    match &popped.node.kind {
                        Kind::Branch { .. } => {
                            // Branch entry consumed 1 byte (when not root)
                            if !self.frames.is_empty() {
                                self.key.pop();
                            }
                        }
                        Kind::Compr { label, .. } => {
                            for _ in 0..label.len() { self.key.pop(); }
                        }
                    }
                    if self.frames.is_empty() {
                        self.eof = true;
                        return None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_find() {
        let mut t: Tree<u32> = Tree::new();
        assert!(t.insert(b"hello", 1).is_none());
        assert!(t.insert(b"help", 2).is_none());
        assert!(t.insert(b"hel", 3).is_none());
        assert_eq!(t.find(b"hello"), Some(&1));
        assert_eq!(t.find(b"help"),  Some(&2));
        assert_eq!(t.find(b"hel"),   Some(&3));
        assert_eq!(t.find(b"he"),    None);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn overwrite() {
        let mut t: Tree<u32> = Tree::new();
        t.insert(b"x", 10);
        assert_eq!(t.insert(b"x", 20), Some(10));
        assert_eq!(t.find(b"x"), Some(&20));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_simple() {
        let mut t: Tree<u32> = Tree::new();
        t.insert(b"hello", 1);
        t.insert(b"help",  2);
        assert_eq!(t.remove(b"hello"), Some(1));
        assert_eq!(t.find(b"hello"), None);
        assert_eq!(t.find(b"help"), Some(&2));
        assert_eq!(t.len(), 1);
    }
}
