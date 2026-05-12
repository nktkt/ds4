//! Integration test for the GGUF parser. Builds a minimal in-memory GGUF
//! payload (header + one metadata kv + one zero-length tensor) and verifies
//! the parser handles it.

use ds4_core::gguf::{Gguf, Value};
use std::io::Write;

fn write_u32(w: &mut Vec<u8>, v: u32) { w.extend_from_slice(&v.to_le_bytes()); }
fn write_u64(w: &mut Vec<u8>, v: u64) { w.extend_from_slice(&v.to_le_bytes()); }
fn write_str(w: &mut Vec<u8>, s: &str) {
    write_u64(w, s.len() as u64);
    w.extend_from_slice(s.as_bytes());
}

fn synth_gguf() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    write_u32(&mut out, 3);           // version
    write_u64(&mut out, 1);           // n_tensors
    write_u64(&mut out, 1);           // n_kv
    // KV: "general.architecture" = "ds4"
    write_str(&mut out, "general.architecture");
    write_u32(&mut out, 8);           // String value kind
    write_str(&mut out, "ds4");
    // Tensor: name="zeros", dims=[0], dtype=F32, offset=0
    write_str(&mut out, "zeros");
    write_u32(&mut out, 1);           // n_dims
    write_u64(&mut out, 0);           // dim 0 = 0 elements
    write_u32(&mut out, 0);           // dtype F32
    write_u64(&mut out, 0);           // tensor offset
    // Pad to alignment 32.
    while out.len() % 32 != 0 { out.push(0); }
    out
}

#[test]
fn parser_round_trips_minimal_gguf() {
    let bytes = synth_gguf();
    let dir = tempdir();
    let path = dir.join("ds4-mini.gguf");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
        f.flush().unwrap();
    }
    let g = Gguf::open(&path).expect("parse synth gguf");
    assert_eq!(g.version, 3);
    assert_eq!(g.tensors.len(), 1);
    assert!(g.tensors.contains_key("zeros"));
    let arch = g.meta_str("general.architecture").unwrap_or("");
    assert_eq!(arch, "ds4");
    if let Some(Value::String(s)) = g.metadata.get("general.architecture") {
        assert_eq!(s, "ds4");
    } else {
        panic!("missing or wrong-typed general.architecture");
    }
    let _ = std::fs::remove_file(&path);
}

fn tempdir() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ds4-rs-test-{}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    p
}
