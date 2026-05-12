//! Radix tree — Rust port of antirez/rax.
//!
//! The original C library packs a node's children, child characters, value
//! pointer, and (optionally) inline leaf bitmap into a single variable-length
//! `unsigned char data[]` allocation behind a 32-bit header. We don't do that
//! here: idiomatic Rust uses owning enums + `Vec<u8>` for child labels and
//! `Vec<Box<Node<V>>>` for child pointers. The user-visible behavior is the
//! same — prefix compression, node splitting on insert, re-compression on
//! delete, ordered iteration — but allocations are bigger and pointer-cheaper.
//!
//! Server code in DwarfStar 4 only needs insert / find / remove and ordered
//! iteration for its tool-call routing tree, so the more exotic APIs from the
//! original (`raxTouch`, `raxDefragIterator`, `raxRandomWalk`, low-level
//! `raxNodeCallback`) are deliberately not implemented yet.

#![allow(clippy::needless_range_loop)]

mod tree;
pub use tree::{Tree, Iter, SeekOp};

// Backwards-compatible aliases so port-level code can use the rax_* / rax
// names from the C API directly.
pub type Rax<V> = Tree<V>;
