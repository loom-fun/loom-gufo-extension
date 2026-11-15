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
  package's projector.
- Prompt cache: by default Gufo's RAM tier names Loom checkpoints, which stay
  on the device (zero copy). With `--cache-disk DIR` the entries become
  portable checkpoint images (full KV pages as append rows; recurrent state,
  carry and the partial page as private state), so Gufo's disk tier persists
  them and a restarted server restores them exactly. The provider claims a package directory
  (`manifest:loom`); `serve`, `prompt`, `chat` and `bench` all accept it.

Options: `--loom-device URI`, `--loom-memory-bytes N`, `--loom-preset NAME`.

## Logits: grammars, logprobs, min-p, penalties

A package that declares `llm.logits` (regenerated with the loom-skills kit
that carries the readback) can return the full logit row of its selections.
`--loom-logits=auto` (default) loads such a package with the capability on;
`on` requires it, `off` never uses it. A request that needs the row (a tool
grammar from `tool_choice` required/named or strict tools, `response_format`
json_schema, `logprobs`, `min_p`, repetition/frequency/presence penalties)
marks its row `"logits": true` for the package, the runner reads the row after
every selecting epoch and chooses the token with Gufo's own sampler, then
feeds it back (`loom_text_embed_decode_batch_tokens`). Everything else, and
every package without the capability, keeps the device path unchanged: the
default epoch and decode speed are the ones of a load without the capability.

- Grammars constrain the model's own tokens, so they are written in the dialect the package speaks: its
  `llm_config.json` (shipped in the package) names the tool-call format and the reasoning markers, and
  `src/loom_dialect.hpp` maps them to Gufo's `ToolSyntax`: `qwen_xml`, `hermes`, `glm`, `gemma4`, `harmony`
  (gpt-oss), `json` (Llama), `pythonic` (LFM2: one `[f(k='v'), g()]` list, keyword arguments) and `atem` (Glimmer:
  one message per call, `to=NAME<|message|><atem:invoke name="NAME">`) have a grammar; `none` has no tool calls and
  its tool requests stay unconstrained as before. Glimmer's automatic tool use (`tool_choice` auto) also stays on
  the device: a call header (`to=NAME`) starts like the answer header (`to=user`), so a lazy grammar cannot tell
  them apart; required, named and strict calls are constrained. Reasoning markers are the model's own (`<think>`, gemma's
  `<|channel>thought`, harmony's analysis channel). Response formats, logprobs, min-p and penalties work for
  every package with the capability. The runner logs what it found at load (`loom: logits on ...`). A package
  without `llm_config.json` is probed for the Hermes JSON dialect.
- A request with a grammar, or with controls the device sampler lacks, runs
  undrafted (drafted packages lose the drafting speed-up for that request
  only). A lazy tool grammar (`tool_choice` auto) stays on the device until a
  call starts, then switches to the host for the rest of the request.
- Logit rows cost device memory: `min(spans per epoch, rows) x vocabulary x 4`
  bytes in the package's `outputs` allocation, only in a logits load.

## Limits

- Containers need `--security-opt seccomp=unconfined` for Loom's GPU memory
  reservation.
- Tests of the runner's host sampling run without a GPU against a fake
  `libloom_text_embed`: `GUFO=<gufo checkout> tests/build_test.sh`.
