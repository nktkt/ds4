//! BPE tokenizer + DeepSeek V4 Flash chat templating.
//!
//! Ported from the tokenizer section of `ds4.c` (search for `ds4_tokenize_text`,
//! `ds4_chat_*`, and the BPE merge tables). The original packs the BPE rank
//! table into a flat array; here we use [`ahash::AHashMap`] so the rest of the
//! port stays readable. Special tokens are stored in a separate ordered
//! [`indexmap::IndexMap`] so we can both look up by string and iterate by id.

use crate::api::Tokens;
use ahash::AHashMap;
use indexmap::IndexMap;

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: Vec<Vec<u8>>,        // id -> bytes
    pub vocab_by_bytes: AHashMap<Vec<u8>, i32>,
    pub special: IndexMap<String, i32>, // name -> id, ordered as in GGUF
    pub merges: AHashMap<(i32, i32), i32>, // (left, right) -> merged id
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
    /// Encode plain text. Mirrors `ds4_tokenize_text`.
    pub fn encode(&self, _text: &str, out: &mut Tokens) {
        // TODO: greedy BPE encoder, matching DS4's byte-level pre-tokenizer.
        // The placeholder leaves `out` untouched so unit tests can still build.
        let _ = out;
    }

    /// Encode an already-rendered chat string. Mirrors
    /// `ds4_tokenize_rendered_chat`.
    pub fn encode_rendered_chat(&self, text: &str, out: &mut Tokens) {
        self.encode(text, out);
    }

    pub fn token_text(&self, id: i32) -> Option<&[u8]> {
        self.vocab.get(id as usize).map(|v| v.as_slice())
    }

    pub fn eos_id(&self) -> i32 { self.eos }
}
