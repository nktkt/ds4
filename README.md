# DwarfStar 4 — Rust port (ds4-rs)

This is a Rust rewrite of [`antirez/ds4`](https://github.com/antirez/ds4): a
DeepSeek V4 Flash specific inference engine. The original is C plus an
Objective-C Metal wrapper plus CUDA C++. The Rust port keeps the same shape:
a narrow public engine boundary (`ds4-core`), a CLI binary (`ds4-cli`), an
HTTP server (`ds4-server`), a benchmark binary (`ds4-bench`), Metal and CUDA
backend crates (`ds4-metal`, `ds4-cuda`), and a port of the radix tree (`rax`)
that the server uses for tool-call routing.

## Layout

```
ds4-rs/
├── Cargo.toml             # workspace manifest
├── crates/
│   ├── rax/               # radix tree (port of rax.c/rax.h)
│   ├── ds4-core/          # model loading, tokenizer, sessions, CPU ref,
│   │                       # backend dispatch (port of ds4.c + ds4.h)
│   ├── ds4-metal/         # Metal runtime + kernel wrappers (port of ds4_metal.m)
│   ├── ds4-cuda/          # CUDA runtime + kernels (port of ds4_cuda.cu)
│   ├── ds4-cli/           # ds4 binary (port of ds4_cli.c + linenoise.c)
│   ├── ds4-server/        # ds4-server binary (port of ds4_server.c)
│   └── ds4-bench/         # ds4-bench binary (port of ds4_bench.c)
├── metal/                 # Metal .metal kernel sources (copied from upstream)
├── tests/                 # integration tests / regression
└── scripts/               # download_model.sh, dir-steering helpers
```

## Status

This is an in-progress mechanical port. Mapping from C concepts to Rust:

| C                                       | Rust                                          |
| --------------------------------------- | --------------------------------------------- |
| `ds4_engine` (opaque struct)            | `ds4_core::Engine`                            |
| `ds4_session`                           | `ds4_core::Session`                           |
| `ds4_tokens` (int * + len + cap)        | `Vec<i32>` (with a wrapper that mirrors the C API where needed) |
| `ds4_backend` enum                      | `ds4_core::Backend` enum                      |
| `ds4_think_mode` enum                   | `ds4_core::ThinkMode` enum                    |
| `raxNode` / `rax`                       | `rax::Tree<V>`                                |
| `linenoise`                             | `rustyline`                                   |
| `mmap` GGUF                             | `memmap2::Mmap`                               |
| pthread workers                         | `std::thread` / `rayon` / `tokio`             |
| Metal Objective-C                       | `objc2-metal` bindings                        |
| CUDA host code                          | `cudarc` (CUDA kernels stay in `.cu` files)   |
| `httpserver.c`-style HTTP               | `tokio` + `hyper`                             |
| `cJSON`-style JSON                      | `serde_json`                                  |

## Build

```sh
cargo build --release            # default backend = Metal on macOS / CUDA on Linux
cargo build --release -p ds4-cli # build only the CLI
cargo build --release --no-default-features --features cpu  # CPU-reference build
```

## License

MIT, same as upstream. See LICENSE.
