//! Sampling primitives. Ported from `ds4.c` sampling helpers (search for
//! `sample_argmax`, `sample_top_k`, `sample_top_p`, `sample_min_p`).

use crate::api::TokenScore;
use rand::Rng;

pub fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v { best = i; best_v = v; }
    }
    best as i32
}

pub fn argmax_excluding(logits: &[f32], excluded_id: i32) -> i32 {
    let mut best: i32 = -1;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if i as i32 == excluded_id { continue; }
        if v > best_v { best = i as i32; best_v = v; }
    }
    best
}

pub fn softmax(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() { *v /= sum; }
    }
}

pub fn top_logprobs(logits: &[f32], k: usize) -> Vec<TokenScore> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut log_z = {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let s: f32 = logits.iter().map(|v| (*v - max).exp()).sum();
        max + s.ln()
    };
    if !log_z.is_finite() { log_z = 0.0; }
    idx.iter().take(k).map(|&i| TokenScore {
        id: i as i32,
        logit: logits[i],
        logprob: logits[i] - log_z,
    }).collect()
}

/// Multi-stage sampler: temperature → top-k → top-p → min-p, exactly as the
/// C `sample_token_*` helpers chain it together. `temperature == 0.0` falls
/// through to argmax.
pub fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    rng: &mut impl Rng,
) -> i32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    for v in logits.iter_mut() { *v /= temperature; }
    softmax(logits);

    // top-k truncation
    if top_k > 0 && (top_k as usize) < logits.len() {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &i in &idx[top_k as usize..] {
            logits[i] = 0.0;
        }
    }
    // min-p cutoff
    if min_p > 0.0 {
        let max = logits.iter().cloned().fold(0.0_f32, f32::max);
        let cutoff = max * min_p;
        for v in logits.iter_mut() { if *v < cutoff { *v = 0.0; } }
    }
    // top-p (nucleus) truncation
    if (0.0..1.0).contains(&top_p) {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut acc = 0.0;
        let mut cut = idx.len();
        for (pos, &i) in idx.iter().enumerate() {
            acc += logits[i];
            if acc >= top_p {
                cut = pos + 1;
                break;
            }
        }
        for &i in &idx[cut..] { logits[i] = 0.0; }
    }
    // Renormalize and draw.
    let sum: f32 = logits.iter().sum();
    if sum <= 0.0 {
        return argmax(logits);
    }
    let mut r: f32 = rng.gen::<f32>() * sum;
    for (i, &p) in logits.iter().enumerate() {
        r -= p;
        if r <= 0.0 { return i as i32; }
    }
    (logits.len() - 1) as i32
}
