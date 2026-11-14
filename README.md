# loom-gufo-extension

Runs [loom-fun](https://huggingface.co/loom-fun) Loom model packages in
[Gufo](https://github.com/gufo-org/gufo) on Strix Halo, as ordinary Gufo text
models: Gufo's server, scheduler, sampler policy, tool/reasoning parsing and
CLI options, with the package's own Loom kernels, chat program, tokenizer and
drafter doing the model work.

```sh
loom-fetch huggingface:loom-fun/ggml-org-Qwen3-4B-loom --output=models/qwen3-4b
gufo serve llm --model models/qwen3-4b --sessions 4
```

## How it fits

- `loom-embed/` is the Loom side: `libloom_text_embed.so`, Loom's text model
  layer (package loading as loom-serve registers it, rows, prefill, packed
  decode epochs, drafted verification, chat rendering, streaming
  detokenization) behind a small C API. It is built inside hrx-system v0.1.0
  by Gufo's `nix build .#loom-runtime` (override `embed` with this directory).
- `src/loom_runner.cpp` is the Gufo side: a `TextModelRunner`. Each Gufo
  session is a Loom row; a Gufo decode batch is one packed Loom epoch; with a
  drafter, decode steps return several verified tokens (Gufo's multi-token
  decode). The provider claims a package directory (`manifest:loom`).

Options: `--loom-device URI`, `--loom-memory-bytes N`, `--loom-preset NAME`.

## Prototype limits

- No prefix reuse between requests: Loom row state stays on device, and the
  cache adapter declares no continuation (every request prefills).
- Sampling runs on device with the package's sampler (temperature, top-k,
  top-p, seed). Gufo's grammar constraints are released, so tool calls and
  JSON are unconstrained (as in loom-serve) and parsed by Gufo; logprobs are
  unavailable. Both need the package to expose logits.
- `serve` only; `gufo prompt`, `chat` and `bench` still expect a GGUF.
- Containers need `--security-opt seccomp=unconfined` for Loom's GPU memory
  reservation.
