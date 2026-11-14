# Loom kernels in Gufo, the Gufo way (prototype)

## What it is

Loom joins Gufo as an ordinary text-model extension. Gufo keeps all of its
own machinery:

- the server and its APIs: OpenAI chat and Responses, Anthropic Messages,
  llama-server routes;
- the scheduler, which decides who prefills and which sessions decode
  together;
- admission, logging, CLI options and tool/reasoning parsing.

A loom-fun package supplies the model work:

- its compiled kernels, run as packed prefill/decode epochs;
- its chat program and tokenizer;
- its drafter, with Loom's on-device verification.

The two meet at Gufo's existing step contract, `TextModelRunner`. Nothing in
Gufo had to learn about Loom's server.

```
gufo serve llm --model <loom package dir>
 └─ Gufo HTTP / scheduler / parsers          (unchanged)
     └─ TextModelRunner: loom extension      (src/loom_runner.cpp)
         └─ libloom_text_embed.so            (Loom text model layer, C API)
             └─ the package's programs + kernels on gfx1151
```

| Gufo calls | Loom does |
|---|---|
| `RenderAndTokenize` | package chat program + tokenizer (`chat_prepare`) |
| `Prefill(state, prompt, offset, n)` | row prefill in prefill-capacity chunks |
| `SelectNext` | the row's pending on-device selection (sampling bound per request) |
| `Advance` | consume the selection (or a forced token) and select the next |
| `DecodeBatch(sessions…)` | one packed epoch over those rows |
| multi-token decode | drafted epoch: propose on device, verify, accept |
| streamed pieces | streaming detokenizer (exact concatenation) |

## Changes

**Gufo:** one small contract addition, API 0.3.15. A model path that is a
directory goes to the provider whose existing `probe` claims it
(`ext::ProbeModelDirectory`), which creates its ordinary step runner
(`create`) with no GGUF reader. `serve` takes that path in
`inference_backend.cpp`. `prompt`, `chat` and `bench` take it through
`runner_cli.cpp`, which runs the server's backend in-process. Packaging adds
`.#loom-runtime`, a Nix package that builds hrx-system v0.1.0 (the release
loom-fun targets) with the embedding library; `withTools = true` adds
loom-serve and loom-fetch for side-by-side comparisons.

**Loom:** no source change. `libloom_text_embed.so` (about 1,000 lines of C
in the extension's `loom-embed/`) is compiled inside Loom's tree from the
existing text model layer:

- package loading via `model_configuration_load`/`resolve` and
  `text_model_allocate`, exactly as loom-serve registers a package;
- rows (`row_begin`, `row_prefill`, `row_decode`), packed `decode_batch`
  over `model_epoch`, drafted `verify_batch` over `model_verify`, and row
  checkpoints (`row_pin`/`row_restore`) for Gufo's prefix cache;
- the package's chat program: rendering, sampling options, streaming output
  classification and tool-call parsing; images through the media projector;
- tokenize, detokenize and a streaming decoder.

It exports only `loom_text_embed_*`.

**Extension:** `loom-gufo-extension`, a community extension. Its manifest
claims `manifest:loom`, declares `--loom-device`, `--loom-memory-bytes` and
`--loom-preset`, and links `loom-text-embed` via pkg-config. The runner
(about 1,600 lines) implements Gufo's `TextModelRunner` and `cache::Adapter`:
Gufo's scheduler, prefix cache, parsers and every API endpoint drive the
package's kernels unchanged.

## Limits (and what removes them)

- **Features that need logits:** `tool_choice` required/named, strict
  schemas, `response_format` enforcement, logprobs, min_p and penalties.
  Packages sample on device and logits never leave it; tools and JSON work
  unconstrained, as in loom-serve. Next step: one Loom ABI addition (logits
  readback or a host-sampling hook) and regenerated packages.
- **No disk tier for the prompt cache.** The RAM tier works on Loom row
  checkpoints, which live on the device.
- **`usage.reasoning_tokens` reads 0** for packages whose native reasoning
  markup is not `<think>` (gpt-oss, Muse-Glimmer): Gufo counts it by
  re-decoding raw tokens. The reasoning text itself is split correctly.
- **Greedy outputs can differ from loom-serve at near-ties.** Loom's results
  depend on batch shape (drafted verify windows, prefill chunks); loom-serve
  itself gives a different greedy haiku for Qwen3-8B under load.
- **Containers need `--security-opt seccomp=unconfined`**, because Loom's GPU
  memory reservation fails under Docker's default profile.

## Evidence (Strix Halo gfx1151, 2026-11-14)

Both servers ran the same package, the same Loom build (hrx-system v0.1.0
via Nix) and identical requests (`bench.py`): greedy, thinking off, three
repetitions, medians. Each server ran alone on the GPU.

| Model | Greedy outputs identical | Decode tok/s (loom-serve / gufo) | Prefill tok/s, ~2.1k-token prompt | 4-stream tok/s | TTFT ms |
|---|---|---|---|---|---|
| LiquidAI/LFM3-1.7B | 5/5 | 164.3 / 168.9 (103%) | 6897 / 6726 (98%) | 587.0 / 574.5 (98%) | 16 / 21 |
| Qwen/Qwen2.5-7B-Instruct | 5/5 | 46.9 / 47.2 (101%) | 2008 / 1961 (98%) | 172.6 / 168.3 (98%) | 50 / 74 |
| Qwen/Qwen3-1.7B | 5/5 | 154.1 / 161.4 (105%) | 6567 / 6206 (94%) | 566.6 / 559.8 (99%) | 17 / 24 |
| Qwen/Qwen3-14B | 5/5 | 24.8 / 25.0 (101%) | 1011 / 989 (98%) | 95.2 / 90.3 (95%) | 103 / 146 |
| Qwen/Qwen3-4B | 5/5 | 97.3 / 96.1 (99%) | 2968 / 2796 (94%) | 262.0 / 248.7 (95%) | 35 / 70 |

Percentages are gufo relative to loom-serve. The higher TTFT on gufo is
expected: the runner consumes the prompt's last token only once Gufo hands it
the request's sampler, which costs one small extra step.

Conformance across all 48 loom-fun packages of 35B parameters or less
(`conformance.py`, 25 checks each, plus images and the CLI) is summarized in
`results/conformance.md`.

The gufo feature suite (`e2e.py`) passed 12/12 through the runner on
Qwen3-4B: chat; streamed reasoning split; tool call, round trip and
streamed tool call; Responses; Anthropic Messages; stop sequences;
4 concurrent requests; decode; and client-disconnect cancellation.
