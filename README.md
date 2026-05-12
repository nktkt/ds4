# ds4 — Rust port of DwarfStar 4

A Rust rewrite of [`antirez/ds4`](https://github.com/antirez/ds4), the
DeepSeek V4 Flash specific inference engine. The upstream project is
~61k lines of C + Objective-C (Metal) + CUDA C++; this port keeps the
same shape but rewrites it as a Cargo workspace.

> **Status: alpha — in-progress mechanical port.** The scaffold compiles
> clean, all three binaries link, 26 unit + integration tests pass. The
> forward-pass / GPU kernels are still being filled in. Not yet usable
> for inference.

## Layout

```
ds4/
├── Cargo.toml             # workspace manifest
├── crates/
│   ├── rax/               # radix tree (port of rax.c/rax.h)
│   ├── ds4-core/          # model loading, tokenizer, sessions, CPU ref,
│   │                      # backend dispatch (port of ds4.c + ds4.h)
│   ├── ds4-metal/         # Metal runtime + kernel wrappers (ds4_metal.m)
│   ├── ds4-cuda/          # CUDA runtime + kernels (ds4_cuda.cu)
│   ├── ds4-cli/           # `ds4` binary (port of ds4_cli.c + linenoise.c)
│   ├── ds4-server/        # `ds4-server` binary (port of ds4_server.c)
│   └── ds4-bench/         # `ds4-bench` binary (port of ds4_bench.c)
├── metal/                 # Metal .metal kernel sources (copied from upstream)
├── tests/                 # integration tests / regression
└── scripts/               # download_model.sh, dir-steering helpers
```

## C → Rust mapping

| C                                       | Rust                                          |
| --------------------------------------- | --------------------------------------------- |
| `ds4_engine` (opaque struct)            | `ds4_core::Engine`                            |
| `ds4_session`                           | `ds4_core::Session`                           |
| `ds4_tokens` (int\* + len + cap)        | `ds4_core::Tokens` (wraps `Vec<i32>`)         |
| `ds4_backend` enum                      | `ds4_core::Backend` enum                      |
| `ds4_think_mode` enum                   | `ds4_core::ThinkMode` enum                    |
| `raxNode` / `rax`                       | `rax::Tree<V>`                                |
| `linenoise`                             | `rustyline`                                   |
| `mmap` GGUF                             | `memmap2::Mmap`                               |
| pthread workers                         | `std::thread` / `rayon` / `tokio`             |
| Metal Objective-C                       | `objc2-metal` bindings                        |
| CUDA host code                          | `cudarc` (CUDA kernels stay in `.cu` files)   |
| hand-rolled epoll/kqueue HTTP           | `tokio` + `hyper`                             |
| `cJSON`-style JSON                      | `serde_json`                                  |

The original `f16_to_f32`, IQ2_XXS lookup tables, RoPE/YaRN math, etc.
are ported with the same bit recipes so logits match upstream
bit-for-bit (cross-checked against the `half` crate in tests).

## What's ported

The smaller, self-contained pieces are already faithful ports with tests:

- **rax**: insert / find / remove / ordered iteration (3 tests)
- **GGUF parser**: header, metadata kv (all 13 value kinds), tensor
  directory, alignment padding, mmap-backed tensor slices
- **IQ2_XXS**: full grid (256 entries) + signed-grid expansion
- **f16 ↔ f32**: matches `half` crate bit-for-bit; E4M3FN microscale dequant
- **RMSNorm** (with/without weights), **per-head RMSNorm**, **SwiGLU**,
  **SiLU**, **softmax**
- **YaRN RoPE** rotation, per-layer base/scale lookup
- **Q8\_0 matvec** (row dot + matvec), **f16 matvec**, **f32 matvec**,
  **Q8\_0 activation quantizer**
- **Q2\_K / Q4\_K / IQ2\_XXS / Q8\_K** dequant block layouts via
  `bytemuck` Pod derives (Q2\_K marked preliminary)
- **BPE tokenizer**: GPT-2 byte map, JoyAI-style pre-tokenizer
  (letters / digits / whitespace / CJK / punct), greedy merge engine,
  special-token splitter
- **Chat encoder**: `encode_chat_prompt`, `append_message`,
  `append_max_effort_prefix`, `append_assistant_prefix`
- **Sampler**: argmax, argmax-excluding, top-k, top-p, min-p, temperature,
  top-logprobs
- **Session sync** / common-prefix / rewrite_from_common
- **Context-memory estimator** matching `ds4_context_memory_estimate`
- **ds4-bench**: faithful argv port (`--ctx-start` / `--ctx-max` /
  `--step-mul` / `--gen-tokens` / `--csv`) + snapshot/restore around
  decode at each frontier
- **ds4-cli**: clap CLI, rustyline REPL, transcript renderer that uses
  the ported chat helpers
- **ds4-server**: tokio + hyper, `/v1/chat/completions` wired end-to-end
  with SSE streaming through a worker queue, `/v1/messages`,
  `/v1/models`, `/healthz`, on-disk KV-cache scaffold, stop-list +
  UTF-8 stream-safe trim

## What's left

The largest chunks remain:

- **Engine forward pass** in `ds4-core`: full attention with MLA
  (multi-head latent attention), MoE routing + shared expert, all
  43 transformer layers
- **Metal backend** (`ds4_metal.m`, 14.6k C lines): real `objc2-metal`
  bindings, MTLBuffer arena, kernel encoders for every DS4 dispatch
- **CUDA backend** (`ds4_cuda.cu`, 10k C lines): `build.rs` invoking
  `nvcc` to compile kernels, host-side launchers via `cudarc`
- **Server JSON parsing**: tool-call shape, on-disk KV cache hit/miss
  policy, Anthropic streaming bodies, reasoning-effort and
  thinking-control fields

## Build

```sh
cargo build --release            # workspace build (host-native backend default)
cargo build --release -p ds4-cli # build only the CLI
cargo test --workspace           # run all tests
```

Backends are gated by Cargo features (`metal`, `cuda`, `cpu`); the
default is platform-native. The CPU backend is currently a stub.

## License

MIT, matching upstream. See `LICENSE`.

## Acknowledgements

- [`antirez/ds4`](https://github.com/antirez/ds4) — the upstream
  DwarfStar 4 engine this port is derived from.
- [`llama.cpp` and GGML](https://github.com/ggml-org/llama.cpp) —
  upstream credits these for kernels, quant formats, GGUF ecosystem,
  and engineering knowledge. We do too.
