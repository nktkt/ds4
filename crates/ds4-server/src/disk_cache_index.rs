//! In-memory index of disk-cache snapshots, keyed by token prefix.
//!
//! Ported from the disk-cache lookup machinery in `ds4_server.c`
//! (`kv_cache_find_text_prefix`, `kv_cache_refresh`, `kv_cache_push` and the
//! `sha1_bytes_hex` / `sha_hex_name` helpers around them). The C server
//! fingerprints the textual prompt with SHA1 and walks the entry array
//! linearly, picking the entry whose stored `text_bytes` is the longest
//! prefix of the incoming prompt. We keep the same semantics — "find the
//! longest cached prefix of the request" — but operate on the *token*
//! sequence directly and store entries in a radix tree (`rax::Tree`) so the
//! longest-prefix walk is O(prefix length) instead of O(num entries).
//!
//! The hashing functions here are intentionally lighter than SHA1: the
//! Rust port only needs a fast 64-bit fingerprint for cheap equality /
//! invalidation checks (CacheMeta.prompt_fingerprint, .model_fingerprint).
//! A SplitMix64-style finalizer driven by a tiny xxhash-like accumulation
//! loop is plenty, deterministic across runs, and dependency-free.

use crate::disk_cache::CacheMeta;
use std::path::Path;

/// SplitMix64 finalizer — the classic Stafford-mixed avalanche step.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Mix a 64-bit chunk into an accumulator. Modeled on the xxhash inner loop
/// but trimmed to a single lane, since the inputs we hash are short.
#[inline]
fn mix(acc: u64, chunk: u64) -> u64 {
    let mut a = acc ^ chunk.wrapping_mul(0xC2B2AE3D27D4EB4F);
    a = a.rotate_left(31).wrapping_mul(0x9E3779B185EBCA87);
    a ^ (a >> 33)
}

/// Fast 64-bit fingerprint of a token slice.
///
/// Mirrors the role of `sha1_bytes_hex(prompt_text, ...)` in
/// `kv_cache_find_text_prefix`: a deterministic content fingerprint used to
/// equality-check a cached snapshot against the live prompt. We hash the
/// token IDs directly (no_std-friendly, no allocation) so two identical
/// token streams always produce the same value across processes and
/// platforms.
pub fn fingerprint_tokens(tokens: &[i32]) -> u64 {
    let mut acc: u64 = 0xCBF29CE484222325 ^ (tokens.len() as u64).wrapping_mul(0x100000001B3);
    for &t in tokens {
        // Sign-extend then zero-extend to a stable 64-bit lane; this keeps
        // negative token IDs (rare but legal) deterministic.
        let lane = (t as i64) as u64;
        acc = mix(acc, lane);
    }
    splitmix64(acc)
}

/// Fingerprint identifying a model file. Combines the path bytes, file size
/// and mtime so any of those changing invalidates cache hits. Mirrors the
/// effect of `kv_cache_existing_compatible`'s checks on the snapshot's
/// stored model identity (quant bits + path identity in C).
pub fn fingerprint_model(model_path: &Path, size: u64, mtime: u64) -> u64 {
    let bytes = model_path.as_os_str().to_string_lossy();
    let mut acc: u64 = 0xCBF29CE484222325 ^ size.wrapping_mul(0x100000001B3);
    // Fold the path bytes 8 at a time, then any remainder.
    let b = bytes.as_bytes();
    let mut i = 0;
    while i + 8 <= b.len() {
        let chunk = u64::from_le_bytes([
            b[i], b[i + 1], b[i + 2], b[i + 3],
            b[i + 4], b[i + 5], b[i + 6], b[i + 7],
        ]);
        acc = mix(acc, chunk);
        i += 8;
    }
    if i < b.len() {
        let mut tail = [0u8; 8];
        tail[..b.len() - i].copy_from_slice(&b[i..]);
        acc = mix(acc, u64::from_le_bytes(tail));
    }
    acc = mix(acc, mtime);
    splitmix64(acc)
}

/// Encode a token sequence as a byte string suitable for use as a radix-tree
/// key. We use unsigned LEB128 (varint) over the token IDs after a zigzag
/// step so positive IDs (the common case, up to 32 bits) compress to 1-5
/// bytes while still being collision-free for any two distinct token
/// sequences. The crucial property the radix tree relies on is *prefix
/// preservation*: if `tokens_a` is a token-prefix of `tokens_b`, then
/// `tokens_to_key(tokens_a)` is a byte-prefix of `tokens_to_key(tokens_b)`.
/// Varint encoding has this property because every token boundary is
/// self-delimiting (the high bit marks continuation).
pub fn tokens_to_key(tokens: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    for &t in tokens {
        // ZigZag fold so negative IDs still produce unsigned varints.
        let zz: u64 = (((t as i64) << 1) ^ ((t as i64) >> 63)) as u64;
        let mut v = zz;
        loop {
            let byte = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                break;
            } else {
                out.push(byte | 0x80);
            }
        }
    }
    out
}

/// Prefix index over disk-cache entries.
///
/// Equivalent in spirit to the in-memory `kv_disk_cache.entry[]` array in
/// the C server, plus the linear `kv_cache_find_text_prefix` scan, but
/// rebuilt as a radix tree on the token sequence. `longest_prefix` returns
/// the deepest visited keyed node, which is the "longest cached prefix of
/// this prompt" — the same answer C computes by picking the entry with the
/// largest matching `text_bytes`.
pub struct PrefixIndex {
    tree: rax::Tree<CacheMeta>,
}

impl Default for PrefixIndex {
    fn default() -> Self { Self::new() }
}

impl PrefixIndex {
    /// Create an empty index. Mirrors `kv_cache_clear` on a fresh
    /// `kv_disk_cache`.
    pub fn new() -> Self {
        Self { tree: rax::Tree::new() }
    }

    /// Insert (or overwrite) the snapshot metadata for a token prefix.
    /// Equivalent to `kv_cache_push` followed by re-sorting in the C
    /// version, except duplicates are simply overwritten here.
    pub fn insert(&mut self, tokens: &[i32], meta: CacheMeta) {
        let key = tokens_to_key(tokens);
        self.tree.insert(&key, meta);
    }

    /// Find the longest cached token prefix of `tokens`. Returns the
    /// matching prefix length in *tokens* (not bytes) and a clone of the
    /// stored metadata. Returns `None` if no stored entry is a prefix of
    /// the input. This is the Rust equivalent of `kv_cache_find_text_prefix`
    /// returning the index of the longest-matching entry.
    pub fn longest_prefix(&self, tokens: &[i32]) -> Option<(u32, CacheMeta)>
    where
        CacheMeta: Clone,
    {
        // Walk token-by-token, querying the radix tree for the cumulative
        // varint key. The radix tree's `find` only returns hits on
        // explicitly inserted keys, which is exactly what we want: only
        // *stored* prefixes are candidates, not arbitrary intermediate
        // token counts. We accumulate the key incrementally to keep this
        // O(prefix_bytes) total rather than O(n^2).
        let mut key: Vec<u8> = Vec::with_capacity(tokens.len() * 2);
        let mut best: Option<(u32, CacheMeta)> = None;
        for (i, &t) in tokens.iter().enumerate() {
            // Append the varint encoding of one token.
            let zz: u64 = (((t as i64) << 1) ^ ((t as i64) >> 63)) as u64;
            let mut v = zz;
            loop {
                let byte = (v & 0x7F) as u8;
                v >>= 7;
                if v == 0 {
                    key.push(byte);
                    break;
                } else {
                    key.push(byte | 0x80);
                }
            }
            if let Some(meta) = self.tree.find(&key) {
                best = Some(((i + 1) as u32, meta.clone()));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_for(session: &str, n_tokens: u32) -> CacheMeta {
        CacheMeta {
            session_id: session.to_string(),
            prompt_fingerprint: fingerprint_tokens(&[1, 2, 3]),
            model_fingerprint: 0,
            n_tokens,
            created_unix: 0,
        }
    }

    #[test]
    fn fingerprint_tokens_is_deterministic() {
        let a = fingerprint_tokens(&[1, 2, 3, 4, 5]);
        let b = fingerprint_tokens(&[1, 2, 3, 4, 5]);
        assert_eq!(a, b);
        // Empty input is also stable.
        assert_eq!(fingerprint_tokens(&[]), fingerprint_tokens(&[]));
    }

    #[test]
    fn fingerprint_tokens_changes_with_input() {
        let base = fingerprint_tokens(&[10, 20, 30]);
        assert_ne!(base, fingerprint_tokens(&[10, 20, 31]));
        assert_ne!(base, fingerprint_tokens(&[10, 20]));
        assert_ne!(base, fingerprint_tokens(&[10, 20, 30, 40]));
        // Order matters.
        assert_ne!(fingerprint_tokens(&[1, 2]), fingerprint_tokens(&[2, 1]));
    }

    #[test]
    fn prefix_index_returns_longest_match() {
        let mut idx = PrefixIndex::new();
        idx.insert(&[1, 2, 3], meta_for("short", 3));
        idx.insert(&[1, 2, 3, 4, 5], meta_for("long", 5));
        let hit = idx.longest_prefix(&[1, 2, 3, 4, 5, 6, 7]);
        let (n, meta) = hit.expect("expected a match");
        assert_eq!(n, 5);
        assert_eq!(meta.session_id, "long");
        // Looking up something that only matches the shorter prefix should
        // return the shorter one.
        let (n2, meta2) = idx
            .longest_prefix(&[1, 2, 3, 9, 9])
            .expect("expected the shorter prefix");
        assert_eq!(n2, 3);
        assert_eq!(meta2.session_id, "short");
    }

    #[test]
    fn prefix_index_no_match_returns_none() {
        let mut idx = PrefixIndex::new();
        idx.insert(&[1, 2, 3], meta_for("s", 3));
        assert!(idx.longest_prefix(&[7, 8, 9]).is_none());
        assert!(idx.longest_prefix(&[]).is_none());
        // A prefix of a stored entry is NOT a stored entry itself — no
        // match, mirroring the "stored snapshot must be a byte-prefix of
        // the prompt" rule in `kv_cache_find_text_prefix`.
        assert!(idx.longest_prefix(&[1, 2]).is_none());
    }
}
