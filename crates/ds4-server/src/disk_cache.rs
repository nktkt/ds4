//! On-disk KV cache. Ported from `ds4_server.c::disk_cache_*`.
//!
//! The on-disk shape mirrors the C version: a directory containing one
//! `<session-id>.kv` file per saved snapshot, plus a small `meta.json` per
//! file describing the prompt fingerprint and creation time. The server keeps
//! a memory index keyed by the prompt token-prefix hash; on hit, we mmap
//! the file and hand the bytes to `Session::load_payload`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMeta {
    pub session_id: String,
    pub prompt_fingerprint: u64,
    pub model_fingerprint: u64,
    pub n_tokens: u32,
    pub created_unix: u64,
}

pub struct DiskCache {
    pub root: PathBuf,
    pub max_bytes: u64,
}

impl DiskCache {
    pub fn new(root: impl Into<PathBuf>, max_bytes: u64) -> Self {
        Self { root: root.into(), max_bytes }
    }

    pub fn path_for(&self, session_id: &str) -> PathBuf {
        self.root.join(format!("{session_id}.kv"))
    }
    pub fn meta_path_for(&self, session_id: &str) -> PathBuf {
        self.root.join(format!("{session_id}.meta.json"))
    }

    pub fn save(&self, meta: &CacheMeta, payload: &[u8]) -> Result<()> {
        std::fs::create_dir_all(&self.root)?;
        std::fs::write(self.path_for(&meta.session_id), payload)?;
        std::fs::write(self.meta_path_for(&meta.session_id), serde_json::to_vec_pretty(meta)?)?;
        self.evict_if_needed()?;
        Ok(())
    }

    pub fn load(&self, session_id: &str) -> Result<(CacheMeta, Vec<u8>)> {
        let meta: CacheMeta = serde_json::from_slice(&std::fs::read(self.meta_path_for(session_id))?)?;
        let payload = std::fs::read(self.path_for(session_id))?;
        Ok((meta, payload))
    }

    pub fn evict_if_needed(&self) -> Result<()> {
        if self.max_bytes == 0 { return Ok(()); }
        // Compute total size, drop oldest snapshots first when over budget.
        let mut entries: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
        for e in std::fs::read_dir(&self.root)?.flatten() {
            let p = e.path();
            if p.extension().map_or(false, |ext| ext == "kv") {
                let md = e.metadata()?;
                let mtime = md.modified().unwrap_or(std::time::UNIX_EPOCH);
                entries.push((p, md.len(), mtime));
            }
        }
        entries.sort_by_key(|(_, _, t)| *t);
        let mut total: u64 = entries.iter().map(|(_, sz, _)| *sz).sum();
        for (p, sz, _) in entries.iter() {
            if total <= self.max_bytes { break; }
            let _ = std::fs::remove_file(p);
            let meta_p = p.with_extension("meta.json");
            let _ = std::fs::remove_file(&meta_p);
            total = total.saturating_sub(*sz);
        }
        Ok(())
    }

    pub fn purge(&self) -> Result<()> {
        if !self.root.exists() { return Ok(()); }
        for e in std::fs::read_dir(&self.root)?.flatten() {
            let _ = std::fs::remove_file(e.path());
        }
        Ok(())
    }
}

pub fn ensure_dir(p: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(p)
}
