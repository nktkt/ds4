//! Byte-level BPE engine.
//!
//! Ports `bpe_emit_piece` / `bpe_rank` / `bpe_tokenize_text` from `ds4.c`.
//! Operates on already-byte-encoded UTF-8 pieces and the merge-rank table.

use super::byte_map::{byte_encode, utf8_len_from_first_byte};
use ahash::AHashMap;

/// One pre-tokenized chunk that the BPE engine merges greedily.
pub struct Bpe<'a> {
    pub merges: &'a AHashMap<(Vec<u8>, Vec<u8>), i32>, // (a,b) → rank
    pub vocab:  &'a AHashMap<Vec<u8>, i32>,            // piece → token id
}

impl<'a> Bpe<'a> {
    /// Tokenize a single piece using the byte-level BPE. Pushes the resulting
    /// token ids onto `out`. Mirrors `bpe_emit_piece`.
    pub fn emit_piece(&self, raw: &[u8], out: &mut Vec<i32>) {
        let encoded = byte_encode(raw);
        // Split into one symbol per UTF-8 char.
        let mut syms: Vec<Vec<u8>> = Vec::new();
        let mut off = 0;
        while off < encoded.len() {
            let mut n = utf8_len_from_first_byte(encoded[off]);
            if off + n > encoded.len() { n = 1; }
            syms.push(encoded[off..off + n].to_vec());
            off += n;
        }
        // Greedy merge.
        loop {
            let mut best_i: Option<usize> = None;
            let mut best_rank = i32::MAX;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&r) = self.merges.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if r < best_rank { best_rank = r; best_i = Some(i); }
                }
            }
            let i = match best_i { Some(i) => i, None => break };
            let mut merged = syms[i].clone();
            merged.extend_from_slice(&syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }
        // Look up each surviving symbol.
        for sym in syms.iter() {
            if let Some(&id) = self.vocab.get(sym) {
                out.push(id);
            } else {
                // Fallback: emit byte-by-byte tokens (every byte should be
                // present in the vocab as its mapped codepoint).
                for j in 0..sym.len() {
                    if let Some(&id) = self.vocab.get(&sym[j..j + 1]) {
                        out.push(id);
                    }
                }
            }
        }
    }
}
