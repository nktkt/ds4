//! Session KV cache + disk payload serialization.
//!
//! Ported from the KV-cache management section of `ds4.c` (search for
//! `kv_*`, `session_save_payload`, `session_load_payload`). The DS4 cache is
//! split into a raw (full-precision) slice and a compressed slice; backends
//! decide how to materialize them in device memory. The on-disk format is
//! versioned with a header that lets the server invalidate stale snapshots
//! after a model upgrade.
//!
//! NOTE: full payload (de)serialization is large enough that we keep it as a
//! TODO scaffold pending the backend port.

use anyhow::{bail, Result};
use std::io::{Read, Write};

const PAYLOAD_MAGIC: [u8; 4] = *b"DS4K";
const PAYLOAD_VERSION: u32 = 1;

#[derive(Debug)]
pub struct PayloadHeader {
    pub version: u32,
    pub model_fingerprint: u64,
    pub backend_id: u32,
    pub n_tokens: u32,
    pub raw_bytes: u64,
    pub comp_bytes: u64,
}

impl PayloadHeader {
    pub fn write(&self, w: &mut dyn Write) -> Result<()> {
        w.write_all(&PAYLOAD_MAGIC)?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.model_fingerprint.to_le_bytes())?;
        w.write_all(&self.backend_id.to_le_bytes())?;
        w.write_all(&self.n_tokens.to_le_bytes())?;
        w.write_all(&self.raw_bytes.to_le_bytes())?;
        w.write_all(&self.comp_bytes.to_le_bytes())?;
        Ok(())
    }
    pub fn read(r: &mut dyn Read) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != PAYLOAD_MAGIC { bail!("kv payload: bad magic"); }
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf4)?; let version = u32::from_le_bytes(buf4);
        if version != PAYLOAD_VERSION { bail!("kv payload: unsupported version {version}"); }
        r.read_exact(&mut buf8)?; let model_fingerprint = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf4)?; let backend_id = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?; let n_tokens = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf8)?; let raw_bytes = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?; let comp_bytes = u64::from_le_bytes(buf8);
        Ok(Self { version, model_fingerprint, backend_id, n_tokens, raw_bytes, comp_bytes })
    }
    pub fn size_on_disk() -> u64 { 4 + 4 + 8 + 4 + 4 + 8 + 8 }
}
