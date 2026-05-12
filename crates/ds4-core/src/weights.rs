//! Layer weight binder — resolve GGUF tensors into typed per-layer byte slices.
//!
//! Ported from the weight binding section of `ds4.c`:
//!
//! * `ds4_layer_weights` (struct, ~line 1987) — the C side's pointer table of
//!   per-layer `ds4_tensor *` handles.
//! * `weights_bind` / `mtp_weights_bind` (~line 2578) — the C `bind_layer_weights`
//!   equivalent: walks `blk.{il}.*.weight` names and stuffs them into the table.
//!   The task spec calls this `bind_layer_weights` / `weights_from_gguf`; both
//!   refer to the same string-to-pointer resolution step.
//! * `tensor_data` (~line 1472) — turns a `ds4_tensor *` into a raw pointer into
//!   the mmap region. On the Rust side this is [`Gguf::tensor_bytes`] plus
//!   `bytemuck::cast_slice` to reinterpret as f32 / u16 / u8.
//! * `weights_validate_layout` (~line 2285) — shape / dtype checks. We mirror
//!   the spirit (mandatory tensors get dtype + element-count validation) without
//!   the full per-dimension audit, which still lives in the upstream C path.
//!
//! Returns `Result<_, String>` so the caller can surface a single human-readable
//! message; this matches the rest of the loader path which fans out into
//! anyhow / context elsewhere but is easy to convert at the boundary.

use bytemuck::cast_slice;

use crate::gguf::{DType, Gguf, Tensor};
use crate::model::Config;
use crate::shape::{
    N_EMBD, N_EXPERT, N_FF_EXP, N_HC, N_HEAD_DIM, N_LAYER, N_VOCAB,
};

/// Per-layer typed view onto the mmap-backed GGUF tensor data.
///
/// Mirrors the C `ds4_layer_weights` struct (ds4.c ~line 1987), but resolved
/// directly to typed byte slices borrowed from the GGUF mmap region instead of
/// going through a `ds4_tensor *` indirection. Fields that may legitimately be
/// absent from a given quant (e.g. routed MoE experts in a dense build) are
/// `Option`, matching the C side's `tensor_by_namef` (which returns NULL) vs
/// `required_tensorf` (which exits).
#[derive(Debug, Clone)]
pub struct LayerWeights<'g> {
    /// Layer index (`il` on the C side).
    pub il: u32,

    // --- Attention ---
    /// `blk.{il}.attn_norm.weight` — F32 RMSNorm scale, length `N_EMBD`.
    pub attn_norm: &'g [f32],
    /// `blk.{il}.attn_q.weight` — Q8_0-packed Q projection bytes.
    pub attn_q: &'g [u8],
    /// `blk.{il}.attn_kv.weight` — Q8_0-packed combined K/V projection bytes.
    pub attn_kv: &'g [u8],
    /// `blk.{il}.attn_output.weight` — Q8_0-packed attention output bytes.
    pub attn_out: &'g [u8],

    // --- FFN (shared expert + optional routed experts) ---
    /// `blk.{il}.ffn_norm.weight` — F32 RMSNorm scale, length `N_EMBD`.
    pub ffn_norm: &'g [f32],
    /// `blk.{il}.ffn_gate_shexp.weight` — Q8_0 shared-expert gate proj.
    pub ffn_gate_shexp: &'g [u8],
    /// `blk.{il}.ffn_up_shexp.weight` — Q8_0 shared-expert up proj.
    pub ffn_up_shexp: &'g [u8],
    /// `blk.{il}.ffn_down_shexp.weight` — Q8_0 shared-expert down proj.
    pub ffn_down_shexp: &'g [u8],
    /// `blk.{il}.ffn_gate_exps.weight` — Q2_K / IQ2_XXS routed gate experts.
    pub ffn_gate_exps: Option<&'g [u8]>,
    /// `blk.{il}.ffn_up_exps.weight` — Q2_K / IQ2_XXS routed up experts.
    pub ffn_up_exps: Option<&'g [u8]>,
    /// `blk.{il}.ffn_down_exps.weight` — Q2_K routed down experts.
    pub ffn_down_exps: Option<&'g [u8]>,
    /// `blk.{il}.ffn_gate_inp.weight` — F16 routing matrix
    /// (`N_EMBD x N_EXPERT`), present only when routed experts are present.
    pub ffn_router: Option<&'g [u16]>,

    // --- Hyper-connection mixers ---
    /// `blk.{il}.hc_attn_fn.weight` — F16 attention HC mixer.
    pub hc_attn_fn: &'g [u16],
    /// `blk.{il}.hc_attn_scale.weight` — F32 attention HC scale (length 3).
    pub hc_attn_scale: &'g [f32],
    /// `blk.{il}.hc_attn_base.weight` — F32 attention HC base.
    pub hc_attn_base: &'g [f32],
    /// `blk.{il}.hc_ffn_fn.weight` — F16 FFN HC mixer.
    pub hc_ffn_fn: &'g [u16],
    /// `blk.{il}.hc_ffn_scale.weight` — F32 FFN HC scale (length 3).
    pub hc_ffn_scale: &'g [f32],
    /// `blk.{il}.hc_ffn_base.weight` — F32 FFN HC base.
    pub hc_ffn_base: &'g [f32],
}

/// Resolved view of every tensor we read during inference.
///
/// Mirrors the C `ds4_weights` struct (ds4.c ~line 2025). The MTP draft head
/// (`ds4_mtp_weights`) lives outside the scope of this slice and is intentionally
/// not exposed here.
#[derive(Debug, Clone)]
pub struct ModelWeights<'g> {
    /// `token_embd.weight` — F16 embedding table, `N_EMBD x N_VOCAB`.
    pub tok_embed: &'g [u16],
    /// `output_norm.weight` — F32 final RMSNorm scale, length `N_EMBD`.
    pub output_norm: &'g [f32],
    /// `output.weight` — Q8_0 vocabulary projection.
    pub output: &'g [u8],
    /// `output_hc_fn.weight` — F16 output HC mixer.
    pub output_hc_fn: &'g [u16],
    /// `output_hc_scale.weight` — F32 output HC scale (length 1).
    pub output_hc_scale: &'g [f32],
    /// `output_hc_base.weight` — F32 output HC base (length `N_HC`).
    pub output_hc_base: &'g [f32],
    pub layers: Vec<LayerWeights<'g>>,
}

/// Resolve every DS4 tensor by name and produce a typed view over the mmap.
///
/// Ported from `weights_bind` in `ds4.c` (~line 2578). The C side validates
/// shape and dtype with `weights_validate_layout`; we do the same checks inline
/// while we have each tensor in hand.
///
/// Reinterpretation is done through `bytemuck::cast_slice`, which only succeeds
/// for properly aligned & sized slices — exactly the safe analogue of the
/// `tensor_data(m, t)` pointer cast on the C side.
pub fn bind_weights<'g>(gguf: &'g Gguf, cfg: &Config) -> Result<ModelWeights<'g>, String> {
    if cfg.n_layers != N_LAYER {
        // We can still proceed; downstream code uses cfg.n_layers, not the
        // shape constant. Just sanity check the value is non-zero.
        if cfg.n_layers == 0 {
            return Err("weights: config reports zero layers".to_string());
        }
    }

    let tok_embed_t = required(gguf, "token_embd.weight")?;
    expect_dtype(tok_embed_t, DType::F16)?;
    expect_elems(tok_embed_t, (N_EMBD as u64) * (N_VOCAB as u64))?;

    let output_norm_t = required(gguf, "output_norm.weight")?;
    expect_dtype(output_norm_t, DType::F32)?;
    expect_elems(output_norm_t, N_EMBD as u64)?;

    let output_t = required(gguf, "output.weight")?;
    expect_dtype(output_t, DType::Q8_0)?;
    expect_elems(output_t, (N_EMBD as u64) * (N_VOCAB as u64))?;

    let output_hc_fn_t = required(gguf, "output_hc_fn.weight")?;
    expect_dtype(output_hc_fn_t, DType::F16)?;

    let output_hc_scale_t = required(gguf, "output_hc_scale.weight")?;
    expect_dtype(output_hc_scale_t, DType::F32)?;
    expect_elems(output_hc_scale_t, 1)?;

    let output_hc_base_t = required(gguf, "output_hc_base.weight")?;
    expect_dtype(output_hc_base_t, DType::F32)?;
    expect_elems(output_hc_base_t, N_HC as u64)?;

    let mut layers = Vec::with_capacity(cfg.n_layers as usize);
    for il in 0..cfg.n_layers {
        layers.push(bind_layer(gguf, il)?);
    }

    Ok(ModelWeights {
        tok_embed: as_u16(gguf, tok_embed_t)?,
        output_norm: as_f32(gguf, output_norm_t)?,
        output: as_u8(gguf, output_t)?,
        output_hc_fn: as_u16(gguf, output_hc_fn_t)?,
        output_hc_scale: as_f32(gguf, output_hc_scale_t)?,
        output_hc_base: as_f32(gguf, output_hc_base_t)?,
        layers,
    })
}

/// Resolve a single `blk.{il}.*` group. Mirrors the body of the `for il` loop
/// in `weights_bind` (`ds4.c` ~line 2587).
fn bind_layer<'g>(gguf: &'g Gguf, il: u32) -> Result<LayerWeights<'g>, String> {
    let name = |suffix: &str| format!("blk.{il}.{suffix}");

    // Norms (F32, length N_EMBD).
    let attn_norm_t = required(gguf, &name("attn_norm.weight"))?;
    expect_dtype(attn_norm_t, DType::F32)?;
    expect_elems(attn_norm_t, N_EMBD as u64)?;

    let ffn_norm_t = required(gguf, &name("ffn_norm.weight"))?;
    expect_dtype(ffn_norm_t, DType::F32)?;
    expect_elems(ffn_norm_t, N_EMBD as u64)?;

    // Q/KV/Out projections — Q8_0 on the DS4 quant we ship.
    let attn_q_t = required(gguf, &name("attn_q.weight"))?;
    expect_dtype(attn_q_t, DType::Q8_0)?;

    let attn_kv_t = required(gguf, &name("attn_kv.weight"))?;
    expect_dtype(attn_kv_t, DType::Q8_0)?;
    // KV is N_EMBD x N_HEAD_DIM in the DS4 fixed shape.
    expect_elems(attn_kv_t, (N_EMBD as u64) * (N_HEAD_DIM as u64))?;

    let attn_out_t = required(gguf, &name("attn_output.weight"))?;
    expect_dtype(attn_out_t, DType::Q8_0)?;

    // Shared FFN — Q8_0.
    let ffn_gate_shexp_t = required(gguf, &name("ffn_gate_shexp.weight"))?;
    expect_dtype(ffn_gate_shexp_t, DType::Q8_0)?;
    expect_elems(ffn_gate_shexp_t, (N_EMBD as u64) * (N_FF_EXP as u64))?;

    let ffn_up_shexp_t = required(gguf, &name("ffn_up_shexp.weight"))?;
    expect_dtype(ffn_up_shexp_t, DType::Q8_0)?;
    expect_elems(ffn_up_shexp_t, (N_EMBD as u64) * (N_FF_EXP as u64))?;

    let ffn_down_shexp_t = required(gguf, &name("ffn_down_shexp.weight"))?;
    expect_dtype(ffn_down_shexp_t, DType::Q8_0)?;
    expect_elems(ffn_down_shexp_t, (N_FF_EXP as u64) * (N_EMBD as u64))?;

    // Routed experts and router — optional, since dense / non-MoE quants omit
    // them (mirrors the `tensor_by_namef` NULL path in C).
    let ffn_gate_exps = optional_expert(gguf, &name("ffn_gate_exps.weight"))?;
    let ffn_up_exps = optional_expert(gguf, &name("ffn_up_exps.weight"))?;
    let ffn_down_exps = optional_expert(gguf, &name("ffn_down_exps.weight"))?;
    let ffn_router = match gguf.tensors.get(&name("ffn_gate_inp.weight")) {
        None => None,
        Some(t) => {
            expect_dtype(t, DType::F16)?;
            expect_elems(t, (N_EMBD as u64) * (N_EXPERT as u64))?;
            Some(as_u16(gguf, t)?)
        }
    };

    // Hyper-connection mixers.
    let hc_attn_fn_t = required(gguf, &name("hc_attn_fn.weight"))?;
    expect_dtype(hc_attn_fn_t, DType::F16)?;
    let hc_attn_scale_t = required(gguf, &name("hc_attn_scale.weight"))?;
    expect_dtype(hc_attn_scale_t, DType::F32)?;
    expect_elems(hc_attn_scale_t, 3)?;
    let hc_attn_base_t = required(gguf, &name("hc_attn_base.weight"))?;
    expect_dtype(hc_attn_base_t, DType::F32)?;

    let hc_ffn_fn_t = required(gguf, &name("hc_ffn_fn.weight"))?;
    expect_dtype(hc_ffn_fn_t, DType::F16)?;
    let hc_ffn_scale_t = required(gguf, &name("hc_ffn_scale.weight"))?;
    expect_dtype(hc_ffn_scale_t, DType::F32)?;
    expect_elems(hc_ffn_scale_t, 3)?;
    let hc_ffn_base_t = required(gguf, &name("hc_ffn_base.weight"))?;
    expect_dtype(hc_ffn_base_t, DType::F32)?;

    Ok(LayerWeights {
        il,
        attn_norm: as_f32(gguf, attn_norm_t)?,
        attn_q: as_u8(gguf, attn_q_t)?,
        attn_kv: as_u8(gguf, attn_kv_t)?,
        attn_out: as_u8(gguf, attn_out_t)?,
        ffn_norm: as_f32(gguf, ffn_norm_t)?,
        ffn_gate_shexp: as_u8(gguf, ffn_gate_shexp_t)?,
        ffn_up_shexp: as_u8(gguf, ffn_up_shexp_t)?,
        ffn_down_shexp: as_u8(gguf, ffn_down_shexp_t)?,
        ffn_gate_exps,
        ffn_up_exps,
        ffn_down_exps,
        ffn_router,
        hc_attn_fn: as_u16(gguf, hc_attn_fn_t)?,
        hc_attn_scale: as_f32(gguf, hc_attn_scale_t)?,
        hc_attn_base: as_f32(gguf, hc_attn_base_t)?,
        hc_ffn_fn: as_u16(gguf, hc_ffn_fn_t)?,
        hc_ffn_scale: as_f32(gguf, hc_ffn_scale_t)?,
        hc_ffn_base: as_f32(gguf, hc_ffn_base_t)?,
    })
}

// --- small helpers ---------------------------------------------------------

/// Look up a mandatory tensor by name (mirrors `required_tensor` in `ds4.c`).
fn required<'g>(gguf: &'g Gguf, name: &str) -> Result<&'g Tensor, String> {
    gguf.tensors
        .get(name)
        .ok_or_else(|| format!("weights: required tensor is missing: {name}"))
}

/// Validate that an optional expert tensor (if present) is a routed-expert
/// quant type. Mirrors `tensor_is_routed_expert_type` in `ds4.c` (~line 2223).
fn optional_expert<'g>(gguf: &'g Gguf, name: &str) -> Result<Option<&'g [u8]>, String> {
    let Some(t) = gguf.tensors.get(name) else { return Ok(None) };
    match t.dtype {
        DType::Q2_K | DType::Q4_K | DType::IQ2_XXS => {}
        other => {
            return Err(format!(
                "weights: tensor `{name}` has dtype {other:?}, expected routed expert quant (Q2_K / Q4_K / IQ2_XXS)"
            ));
        }
    }
    Ok(Some(as_u8(gguf, t)?))
}

fn expect_dtype(t: &Tensor, want: DType) -> Result<(), String> {
    if t.dtype != want {
        return Err(format!(
            "weights: tensor `{}` has dtype {:?}, expected {:?}",
            t.name, t.dtype, want
        ));
    }
    Ok(())
}

fn expect_elems(t: &Tensor, want: u64) -> Result<(), String> {
    let got: u64 = t.shape.iter().product();
    if got != want {
        return Err(format!(
            "weights: tensor `{}` has {} elements, expected {}",
            t.name, got, want
        ));
    }
    Ok(())
}

fn as_u8<'g>(gguf: &'g Gguf, t: &Tensor) -> Result<&'g [u8], String> {
    gguf.tensor_bytes(t).map_err(|e| format!("{e:#}"))
}

fn as_u16<'g>(gguf: &'g Gguf, t: &Tensor) -> Result<&'g [u16], String> {
    let bytes = as_u8(gguf, t)?;
    if bytes.len() % std::mem::size_of::<u16>() != 0 {
        return Err(format!(
            "weights: tensor `{}` byte length {} not a multiple of 2 (u16 cast)",
            t.name,
            bytes.len()
        ));
    }
    Ok(cast_slice::<u8, u16>(bytes))
}

fn as_f32<'g>(gguf: &'g Gguf, t: &Tensor) -> Result<&'g [f32], String> {
    let bytes = as_u8(gguf, t)?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!(
            "weights: tensor `{}` byte length {} not a multiple of 4 (f32 cast)",
            t.name,
            bytes.len()
        ));
    }
    Ok(cast_slice::<u8, f32>(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;
    use std::io::Write;

    fn write_u32(w: &mut Vec<u8>, v: u32) { w.extend_from_slice(&v.to_le_bytes()); }
    fn write_u64(w: &mut Vec<u8>, v: u64) { w.extend_from_slice(&v.to_le_bytes()); }
    fn write_str(w: &mut Vec<u8>, s: &str) {
        write_u64(w, s.len() as u64);
        w.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal but plausible GGUF byte payload. Only the metadata key
    /// `general.architecture` is set, and exactly one zero-length F32 tensor
    /// is declared, so the binder can locate it and observe missing tensors
    /// for everything else. Pattern lifted from `tests/gguf_parser.rs`.
    fn synth_gguf(tensor_name: &str) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        write_u32(&mut out, 3);
        write_u64(&mut out, 1); // n_tensors
        write_u64(&mut out, 1); // n_kv
        write_str(&mut out, "general.architecture");
        write_u32(&mut out, 8); // string kind
        write_str(&mut out, "ds4");
        write_str(&mut out, tensor_name);
        write_u32(&mut out, 1); // n_dims
        write_u64(&mut out, 0); // dim0 = 0 elements (size_bytes = 0)
        write_u32(&mut out, 0); // dtype F32
        write_u64(&mut out, 0); // tensor offset
        while out.len() % 32 != 0 { out.push(0); }
        out
    }

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ds4-rs-weights-test-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// The binder must report every missing tensor as a clean `Err(String)`
    /// rather than panicking. We feed it a GGUF that only contains a single
    /// unrelated tensor and check the error names the first missing one.
    #[test]
    fn binder_reports_missing_tensors_cleanly() {
        let bytes = synth_gguf("zeros");
        let dir = tempdir();
        let path = dir.join("ds4-weights-missing.gguf");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }
        let g = Gguf::open(&path).expect("parse synth gguf");

        // We don't have a real Config available, so synthesize a minimal one
        // that won't trip the n_layers==0 guard.
        let cfg = Config {
            n_layers: 1,
            d_model: N_EMBD,
            n_heads: 64,
            n_kv_heads: 1,
            d_head: N_HEAD_DIM,
            d_ff: N_FF_EXP,
            n_experts: N_EXPERT,
            n_experts_per_tok: 6,
            vocab_size: N_VOCAB,
            rope_base: 10_000.0,
            rms_norm_eps: 1.0e-6,
            max_context: 65_536,
            q_lora_rank: 1024,
            kv_lora_rank: 0,
        };

        let err = bind_weights(&g, &cfg).expect_err("should report missing tensors");
        assert!(err.contains("missing"), "error should mention missing tensor: {err}");
        assert!(err.contains("token_embd"), "first failure should be the embedding table: {err}");

        let _ = std::fs::remove_file(&path);
    }
}
