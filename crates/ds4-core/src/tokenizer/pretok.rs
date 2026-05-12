//! Pre-tokenizer. Mirrors the JoyAI/regex pre-tokenizer in `ds4.c`
//! (`joyai_*` helpers + `tokenize_span`). The DS4 tokenizer is byte-level BPE
//! but it pre-splits input into letter / digit / whitespace / punctuation /
//! CJK runs first so very common token shapes survive intact.

pub fn is_ascii_alpha(c: u8) -> bool { c.is_ascii_alphabetic() }
pub fn is_ascii_digit(c: u8) -> bool { c.is_ascii_digit() }
pub fn is_ascii_space(c: u8) -> bool { matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) }
pub fn is_ascii_newline(c: u8) -> bool { matches!(c, b'\n' | b'\r') }
pub fn is_ascii_punct(c: u8) -> bool {
    matches!(c, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~')
}

/// Peek one UTF-8 codepoint at `pos`. Returns (codepoint, next_pos).
pub fn utf8_peek_one(s: &[u8], pos: usize) -> (u32, usize) {
    let c0 = s[pos];
    let mut n = super::byte_map::utf8_len_from_first_byte(c0);
    if pos + n > s.len() { n = 1; }
    let cp = match n {
        1 => c0 as u32,
        2 => ((c0 as u32 & 0x1f) << 6) | (s[pos + 1] as u32 & 0x3f),
        3 => ((c0 as u32 & 0x0f) << 12) | ((s[pos + 1] as u32 & 0x3f) << 6) | (s[pos + 2] as u32 & 0x3f),
        _ => ((c0 as u32 & 0x07) << 18)
            | ((s[pos + 1] as u32 & 0x3f) << 12)
            | ((s[pos + 2] as u32 & 0x3f) << 6)
            | (s[pos + 3] as u32 & 0x3f),
    };
    (cp, pos + n)
}

pub fn is_cjk_or_kana(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)        // CJK Unified Ideographs
        || (0x3040..=0x309f).contains(&cp)  // Hiragana
        || (0x30a0..=0x30ff).contains(&cp)  // Katakana
}

/// Treat non-ASCII non-control bytes as letter-like (mirrors the JoyAI
/// `joyai_letter_like_at` comment). CJK ranges are isolated above before
/// the generic letter rule runs.
pub fn is_letter_like(s: &[u8], pos: usize) -> bool {
    let c = s[pos];
    if c < 128 { return is_ascii_alpha(c); }
    true
}

/// Next UTF-8 byte boundary after `pos`.
pub fn next_utf8(s: &[u8], pos: usize) -> usize {
    let mut n = super::byte_map::utf8_len_from_first_byte(s[pos]);
    if pos + n > s.len() { n = 1; }
    pos + n
}

/// Yield pre-tokenizer spans. Mirrors the run-splitting in `tokenize_span`:
/// letters, digits, whitespace, CJK, and punctuation are emitted as
/// separate spans of contiguous same-category bytes.
pub fn split_spans(s: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < s.len() {
        let start = pos;
        // CJK chars are isolated one codepoint at a time.
        if s[pos] >= 0x80 {
            let (cp, next) = utf8_peek_one(s, pos);
            if is_cjk_or_kana(cp) {
                out.push(&s[start..next]);
                pos = next;
                continue;
            }
        }
        let c = s[pos];
        if is_ascii_digit(c) {
            while pos < s.len() && is_ascii_digit(s[pos]) { pos += 1; }
        } else if is_ascii_space(c) {
            while pos < s.len() && is_ascii_space(s[pos]) { pos += 1; }
        } else if is_letter_like(s, pos) {
            while pos < s.len() && is_letter_like(s, pos) {
                // single-codepoint advance
                pos = next_utf8(s, pos);
            }
        } else {
            // Treat everything else as a single byte.
            pos = next_utf8(s, pos);
        }
        out.push(&s[start..pos]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ascii_split() {
        let s: &[u8] = b"hello world 123";
        let parts = split_spans(s);
        assert_eq!(parts, vec![&b"hello"[..], &b" "[..], &b"world"[..], &b" "[..], &b"123"[..]]);
    }
    #[test]
    fn cjk_each_char_isolated() {
        // 日本 — two CJK ideographs, each 3 UTF-8 bytes
        let s = "日本".as_bytes();
        let parts = split_spans(s);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "日".as_bytes());
        assert_eq!(parts[1], "本".as_bytes());
    }
}
