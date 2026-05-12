//! `ds4_tokens` helpers. The actual `Tokens` type lives in [`crate::api`];
//! this module holds free functions that mirror the C helpers verbatim.

use crate::api::Tokens;

#[inline]
pub fn push(tv: &mut Tokens, token: i32) { tv.push(token); }

#[inline]
pub fn free(tv: &mut Tokens) { tv.clear(); }

#[inline]
pub fn copy(dst: &mut Tokens, src: &Tokens) { dst.copy_from(src); }

#[inline]
pub fn starts_with(tokens: &Tokens, prefix: &Tokens) -> bool {
    tokens.starts_with(prefix)
}

/// Compute the length of the longest common prefix between two token slices.
/// Useful for the session cache logic that reuses a KV prefix.
pub fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
