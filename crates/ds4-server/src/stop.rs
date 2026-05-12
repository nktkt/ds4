//! Stop-list + UTF-8 stream-safe length. Ports `stop_list_*` and
//! `utf8_stream_safe_len` from `ds4_server.c`. Used by the streaming endpoints
//! so we don't emit an SSE delta that ends mid-character or that contains
//! the first half of a stop sequence.

#[derive(Debug, Default, Clone)]
pub struct StopList {
    pub items: Vec<String>,
    pub max_len: usize,
}

impl StopList {
    pub fn push(&mut self, s: String) {
        if s.is_empty() { return; }
        if s.len() > self.max_len { self.max_len = s.len(); }
        self.items.push(s);
    }
    pub fn clear(&mut self) {
        self.items.clear();
        self.max_len = 0;
    }
    pub fn from_iter(it: impl IntoIterator<Item = String>) -> Self {
        let mut out = Self::default();
        for s in it { out.push(s); }
        out
    }
    /// Find the earliest stop match in `text[from..]`. Mirrors
    /// `stop_list_find_from`. Returns `(absolute_pos, match_len)`.
    pub fn find_from(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        if self.items.is_empty() { return None; }
        let mut best: Option<(usize, usize)> = None;
        for s in &self.items {
            if let Some(local) = text[from..].find(s.as_str()) {
                let abs = from + local;
                if best.map_or(true, |(p, _)| abs < p) {
                    best = Some((abs, s.len()));
                }
            }
        }
        best
    }

    /// Streaming cannot emit the last `max_len - 1` bytes yet: a stop
    /// sequence may start there and finish in the next token.
    pub fn stream_safe_len(&self, text_len: usize) -> usize {
        if self.items.is_empty() || self.max_len <= 1 { return text_len; }
        let hold = self.max_len - 1;
        text_len.saturating_sub(hold)
    }
}

/// UTF-8 expected sequence length given the lead byte.
pub fn utf8_expected_len(c: u8) -> usize {
    if c < 0x80 { 1 }
    else if (0xc2..=0xdf).contains(&c) { 2 }
    else if (0xe0..=0xef).contains(&c) { 3 }
    else if (0xf0..=0xf4).contains(&c) { 4 }
    else { 1 }
}

/// Return a byte length that yields a clean UTF-8 cut at or before `limit`,
/// over the buffer `s[start..]`. `final_chunk = true` skips the trim because
/// no further tokens will arrive. Mirrors `utf8_stream_safe_len`.
pub fn utf8_stream_safe_len(s: &[u8], start: usize, limit: usize, final_chunk: bool) -> usize {
    if final_chunk || limit <= start { return limit; }
    // Walk backwards over continuation bytes.
    let mut p = limit;
    let mut cont = 0;
    while p > start && cont < 4 && (s[p - 1] & 0xc0) == 0x80 {
        p -= 1;
        cont += 1;
    }
    if p == limit {
        return if utf8_expected_len(s[limit - 1]) > 1 { limit - 1 } else { limit };
    }
    if p == start && (s[p] & 0xc0) == 0x80 { return start; }
    let lead = p - 1;
    let need = utf8_expected_len(s[lead]);
    if (limit - lead) < need { lead } else { limit }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_first_match() {
        let mut sl = StopList::default();
        sl.push("stop".into());
        sl.push("END".into());
        let text = "hello stop world END";
        assert_eq!(sl.find_from(text, 0), Some((6, 4)));
    }

    #[test]
    fn stream_safe_holds_partial_stop() {
        let mut sl = StopList::default();
        sl.push("stop".into()); // max_len = 4 → hold 3
        assert_eq!(sl.stream_safe_len(10), 7);
        assert_eq!(sl.stream_safe_len(2), 0);
    }

    #[test]
    fn utf8_trims_partial_multibyte() {
        // 'é' = 0xC3 0xA9. If limit is at the first byte only, we should
        // trim back.
        let s = b"hi\xC3\xA9";
        // Pass limit=3 (just the lead byte of é), final=false → must trim
        assert_eq!(utf8_stream_safe_len(s, 0, 3, false), 2);
        // Pass limit=4 (whole é), final=false → no trim
        assert_eq!(utf8_stream_safe_len(s, 0, 4, false), 4);
        // final=true keeps even partial bytes (caller flushes everything)
        assert_eq!(utf8_stream_safe_len(s, 0, 3, true), 3);
    }
}
