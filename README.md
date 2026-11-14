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

## Build

Gufo with this extension needs the model-directory support and the
`loom-runtime` Nix package on the `loom-extension` branch of
[loom-fun/gufo](https://github.com/loom-fun/gufo/tree/loom-extension):

```sh
nix build --impure --expr '((builtins.getFlake "github:loom-fun/gufo/loom-extension").packages.x86_64-linux.loom-runtime.override { embed = ./loom-embed; })'
PKG_CONFIG_PATH=<that output>/lib/pkgconfig cmake <gufo> -DENGINE_ENABLE_HIP=ON \
  -DGUFO_EXTENSIONS=loom -DGUFO_EXTENSION_DIRS=$PWD
```

Design and evidence: `DESIGN.md`; validation clients and results: `validation/`.

## How it fits

- `loom-embed/` is the Loom side: `libloom_text_embed.so`, Loom's text model
  layer (package loading as loom-serve registers it, rows, prefill, packed
  decode epochs, drafted verification, chat rendering, streaming
  detokenization) behind a small C API. It is built inside hrx-system v0.1.0
  by Gufo's `nix build .#loom-runtime` (override `embed` with this directory).
- `src/loom_runner.cpp` is the Gufo side: a `TextModelRunner`. Each Gufo
  session is a Loom row; a Gufo decode batch is one packed Loom epoch; with a
  drafter, decode steps return several verified tokens (Gufo's multi-token
  decode). Row checkpoints back Gufo's prefix cache; images run through the
  package's projector. The provider claims a package directory
  (`manifest:loom`); `serve`, `prompt`, `chat` and `bench` all accept it.

Options: `--loom-device URI`, `--loom-memory-bytes N`, `--loom-preset NAME`.

## Limits

- Needs logits from Loom (not exposed by packages yet): `tool_choice`
  required/named, strict schemas, `response_format` enforcement, logprobs,
  min_p and penalties. Tools and JSON work unconstrained, as in loom-serve.
- The prompt cache has a RAM tier (Loom row checkpoints) but no disk tier.
- `usage.reasoning_tokens` reads 0 for packages whose native reasoning markup
  is not `<think>` (gpt-oss, Muse-Glimmer).
- Containers need `--security-opt seccomp=unconfined` for Loom's GPU memory
  reservation.
