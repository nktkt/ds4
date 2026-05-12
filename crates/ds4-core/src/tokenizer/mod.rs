//! BPE tokenizer + DeepSeek V4 Flash chat templating.
//!
//! Ported from the tokenizer section of `ds4.c`. Sub-modules:
//!
//! * [`byte_map`] — GPT-2 byte ↔ codepoint mapping
//! * [`pretok`]   — JoyAI-style pre-tokenizer (letters / digits / space / CJK)
//! * [`bpe`]      — greedy merge engine on byte-encoded pieces

pub mod byte_map;
pub mod pretok;
pub mod bpe;

use crate::api::Tokens;
use ahash::AHashMap;
use indexmap::IndexMap;

#[derive(Debug, Default)]
pub struct Tokenizer {
    pub vocab: Vec<Vec<u8>>,                // id -> byte-encoded form
    pub vocab_by_bytes: AHashMap<Vec<u8>, i32>,
    pub special: IndexMap<String, i32>,
    pub merges: AHashMap<(Vec<u8>, Vec<u8>), i32>,
    pub bos: i32,
    pub eos: i32,
    pub im_start: i32,
    pub im_end: i32,
    pub user_role: i32,
    pub assistant_role: i32,
    pub system_role: i32,
    pub think_start: i32,
    pub think_end: i32,
}

impl Tokenizer {
    /// Encode plain text. Mirrors `ds4_tokenize_text` / `bpe_tokenize_text`.
    /// The pipeline is:
    ///
    ///   text → pre-tokenizer spans → byte-encode → BPE → token ids
    pub fn encode(&self, text: &str, out: &mut Tokens) {
        if self.vocab.is_empty() { return; }
        let bpe = bpe::Bpe { merges: &self.merges, vocab: &self.vocab_by_bytes };
        let mut buf: Vec<i32> = Vec::new();
        for piece in pretok::split_spans(text.as_bytes()) {
            bpe.emit_piece(piece, &mut buf);
        }
        out.extend(&buf);
    }

    /// Encode an already-rendered chat string. Mirrors
    /// `tokenize_rendered_chat`: splits on special-token boundaries, then
    /// runs BPE on each non-special slice.
    pub fn encode_rendered_chat(&self, text: &str, out: &mut Tokens) {
        if self.special.is_empty() {
            self.encode(text, out);
            return;
        }
        let bytes = text.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            let mut matched: Option<(usize, i32)> = None;
            for (name, &id) in self.special.iter() {
                if bytes.len() - pos >= name.len() && &bytes[pos..pos + name.len()] == name.as_bytes() {
                    matched = Some((name.len(), id));
                    break;
                }
            }
            if let Some((skip, id)) = matched {
                out.push(id);
                pos += skip;
                continue;
            }
            let next = next_special_pos(self, bytes, pos);
            let run = &bytes[pos..next];
            let bpe = bpe::Bpe { merges: &self.merges, vocab: &self.vocab_by_bytes };
            let mut buf: Vec<i32> = Vec::new();
            for piece in pretok::split_spans(run) { bpe.emit_piece(piece, &mut buf); }
            out.extend(&buf);
            pos = next;
        }
    }

    pub fn token_text(&self, id: i32) -> Option<&[u8]> {
        self.vocab.get(id as usize).map(|v| v.as_slice())
    }

    pub fn eos_id(&self) -> i32 { self.eos }
}

fn next_special_pos(t: &Tokenizer, bytes: &[u8], start: usize) -> usize {
    let mut p = start + 1;
    while p < bytes.len() {
        for (name, _) in t.special.iter() {
            if bytes.len() - p >= name.len() && &bytes[p..p + name.len()] == name.as_bytes() {
                return p;
            }
        }
        p += byte_map::utf8_len_from_first_byte(bytes[p]).max(1);
    }
    bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tok() -> Tokenizer {
        let mut t = Tokenizer::default();
        for b in 0u8..128 {
            let bytes = vec![byte_map::byte_to_codepoint(b) as u8];
            t.vocab.push(bytes.clone());
            t.vocab_by_bytes.insert(bytes, b as i32);
        }
        while t.vocab.len() < 200 { t.vocab.push(Vec::new()); }
        t.vocab.push(b"ab".to_vec());
        t.vocab_by_bytes.insert(b"ab".to_vec(), 200);
        t.merges.insert((b"a".to_vec(), b"b".to_vec()), 0);
        t
    }

    #[test]
    fn encodes_merge() {
        let t = make_tok();
        let mut out = Tokens::new();
        t.encode("ab", &mut out);
        assert_eq!(out.as_slice(), &[200]);
    }
}
