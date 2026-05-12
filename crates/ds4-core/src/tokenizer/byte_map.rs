//! GPT-2 byte ↔ codepoint mapping.
//!
//! Ports `gpt2_byte_to_codepoint`, `utf8_put`, and `byte_encode` from
//! `ds4.c`. The byte-level BPE scheme maps every input byte to a printable
//! Unicode codepoint before merging so the merge alphabet is plain UTF-8 and
//! we never lose byte identity.

#[inline]
pub fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0x80 { 1 }
    else if c & 0xe0 == 0xc0 { 2 }
    else if c & 0xf0 == 0xe0 { 3 }
    else if c & 0xf8 == 0xf0 { 4 }
    else { 1 }
}

/// Append the UTF-8 encoding of `cp` to `out`.
pub fn utf8_put(out: &mut Vec<u8>, cp: u32) {
    if cp <= 0x7f {
        out.push(cp as u8);
    } else if cp <= 0x7ff {
        out.push(0xc0 | (cp >> 6) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    } else if cp <= 0xffff {
        out.push(0xe0 | (cp >> 12) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3f) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    } else {
        out.push(0xf0 | (cp >> 18) as u8);
        out.push(0x80 | ((cp >> 12) & 0x3f) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3f) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    }
}

/// `gpt2_byte_to_codepoint` from `ds4.c`. The printable-ASCII bytes map to
/// themselves; everything else gets shifted into the 256..324 codepoint range.
pub fn byte_to_codepoint(b: u8) -> u32 {
    if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
        return b as u32;
    }
    let mut n: u32 = 0;
    for x in 0u32..256 {
        let xb = x as u8;
        if (33..=126).contains(&xb) || (161..=172).contains(&xb) || xb >= 174 { continue; }
        if xb == b { return 256 + n; }
        n += 1;
    }
    b as u32
}

/// Encode `bytes` using the GPT-2 byte mapping. Mirrors `byte_encode`.
pub fn byte_encode(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        utf8_put(&mut out, byte_to_codepoint(b));
    }
    out
}

/// Inverse of `byte_to_codepoint`. Materialized as a 320-entry table so token
/// → raw byte decoding is O(1) per codepoint.
pub fn codepoint_to_byte_table() -> [Option<u8>; 320] {
    let mut t = [None; 320];
    for b in 0u32..256 {
        let cp = byte_to_codepoint(b as u8) as usize;
        if cp < 320 { t[cp] = Some(b as u8); }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trip() {
        for b in 33u8..=126 {
            assert_eq!(byte_to_codepoint(b), b as u32);
        }
    }
    #[test]
    fn nul_maps_high() {
        // byte 0 is non-printable, so it should be remapped to codepoint >= 256
        assert!(byte_to_codepoint(0) >= 256);
    }
    #[test]
    fn encode_smoke() {
        let s = b"Hello";
        let out = byte_encode(s);
        // Each printable ASCII maps to itself, so output equals input here.
        assert_eq!(out, s);
    }
}
