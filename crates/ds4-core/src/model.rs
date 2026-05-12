//! Model config + weight handles.
//!
//! Ported from the `model_*` and tensor lookup section of `ds4.c`. The DS4
//! architecture is hard-coded: MoE with shared experts, multi-head latent
//! attention, RoPE, optional MTP draft tokens. This module turns the GGUF
//! key/value bag into a strongly typed [`Config`] so backends don't have to
//! string-match metadata at runtime.

use anyhow::{anyhow, Context, Result};
use crate::gguf::{Gguf, Tensor};

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub d_head: u32,
    pub d_ff: u32,
    pub n_experts: u32,
    pub n_experts_per_tok: u32,
    pub vocab_size: u32,
    pub rope_base: f32,
    pub rms_norm_eps: f32,
    pub max_context: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let prefix = g.meta_str("general.architecture")
            .ok_or_else(|| anyhow!("model: missing general.architecture"))?
            .to_owned();
        let k = |name: &str| format!("{prefix}.{name}");
        Ok(Self {
            n_layers: g.meta_u32(&k("block_count"))
                .context("model: missing block_count")?,
            d_model:  g.meta_u32(&k("embedding_length")).context("embedding_length")?,
            n_heads:  g.meta_u32(&k("attention.head_count")).context("head_count")?,
            n_kv_heads: g.meta_u32(&k("attention.head_count_kv"))
                .unwrap_or_else(|| g.meta_u32(&k("attention.head_count")).unwrap_or(0)),
            d_head:   g.meta_u32(&k("attention.key_length")).unwrap_or(128),
            d_ff:     g.meta_u32(&k("feed_forward_length")).context("feed_forward_length")?,
            n_experts: g.meta_u32(&k("expert_count")).unwrap_or(0),
            n_experts_per_tok: g.meta_u32(&k("expert_used_count")).unwrap_or(0),
            vocab_size: g.meta_u32(&k("vocab_size"))
                .or_else(|| g.meta_u32("tokenizer.ggml.vocab_size"))
                .unwrap_or(0),
            rope_base: g.meta_f32(&k("rope.freq_base")).unwrap_or(10000.0),
            rms_norm_eps: g.meta_f32(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5),
            max_context: g.meta_u32(&k("context_length")).unwrap_or(0),
            q_lora_rank: g.meta_u32(&k("attention.q_lora_rank")).unwrap_or(0),
            kv_lora_rank: g.meta_u32(&k("attention.kv_lora_rank")).unwrap_or(0),
        })
    }
}

/// A typed lookup of the named tensors we care about. The DS4 graph references
/// a fixed roster of tensor names; the loader resolves them eagerly so any
/// missing tensor surfaces as an error at open time, not at first forward pass.
#[derive(Debug, Clone)]
pub struct Tensors<'g> {
    pub tok_embed: &'g Tensor,
    pub output_norm: &'g Tensor,
    pub output: &'g Tensor,
    pub layers: Vec<LayerTensors<'g>>,
}

#[derive(Debug, Clone)]
pub struct LayerTensors<'g> {
    pub attn_norm: &'g Tensor,
    pub attn_q: &'g Tensor,
    pub attn_kv: &'g Tensor,
    pub attn_out: &'g Tensor,
    pub ffn_norm: &'g Tensor,
    pub ffn_gate: &'g Tensor,
    pub ffn_up: &'g Tensor,
    pub ffn_down: &'g Tensor,
    pub experts_gate: Option<&'g Tensor>,
    pub experts_up: Option<&'g Tensor>,
    pub experts_down: Option<&'g Tensor>,
    pub experts_router: Option<&'g Tensor>,
}

impl<'g> Tensors<'g> {
    pub fn resolve(g: &'g Gguf, cfg: &Config) -> Result<Self> {
        let get = |name: &str| -> Result<&'g Tensor> {
            g.tensors.get(name)
                .ok_or_else(|| anyhow!("model: missing tensor `{name}`"))
        };
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers {
            let lname = |suffix: &str| format!("blk.{i}.{suffix}");
            let opt = |n: &str| g.tensors.get(&lname(n));
            layers.push(LayerTensors {
                attn_norm: get(&lname("attn_norm.weight"))?,
                attn_q:    get(&lname("attn_q.weight"))?,
                attn_kv:   get(&lname("attn_kv.weight"))?,
                attn_out:  get(&lname("attn_output.weight"))?,
                ffn_norm:  get(&lname("ffn_norm.weight"))?,
                ffn_gate:  get(&lname("ffn_gate.weight"))?,
                ffn_up:    get(&lname("ffn_up.weight"))?,
                ffn_down:  get(&lname("ffn_down.weight"))?,
                experts_gate: opt("ffn_gate_exps.weight"),
                experts_up:   opt("ffn_up_exps.weight"),
                experts_down: opt("ffn_down_exps.weight"),
                experts_router: opt("ffn_gate_inp.weight"),
            });
        }
        Ok(Self {
            tok_embed: get("token_embd.weight")?,
            output_norm: get("output_norm.weight")?,
            output: get("output.weight")?,
            layers,
        })
    }
}
