//! Per-layer helpers — currently just `compress_ratio`. Ported from the C
//! `ds4_layer_compress_ratio` near the GGUF helpers.

use crate::shape::N_LAYER;

/// Attention compression ratio for layer `il`. Mirrors `ds4_layer_compress_ratio`:
///
/// * Layer 0 / 1 → dense (ratio 0 → no compression),
/// * Even layers from 2 onward → ratio 4 (compressed, with indexer),
/// * Odd layers from 3 onward → ratio 128 (compressed, no indexer).
pub fn compress_ratio(il: u32) -> u32 {
    assert!(il < N_LAYER, "DeepSeek4 layer index outside fixed layout");
    if il < 2 { return 0; }
    if il & 1 == 0 { 4 } else { 128 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_compression_pattern() {
        assert_eq!(compress_ratio(0), 0);
        assert_eq!(compress_ratio(1), 0);
        assert_eq!(compress_ratio(2), 4);
        assert_eq!(compress_ratio(3), 128);
        assert_eq!(compress_ratio(4), 4);
        assert_eq!(compress_ratio(5), 128);
        assert_eq!(compress_ratio(42), 4); // 42 even
    }
}
