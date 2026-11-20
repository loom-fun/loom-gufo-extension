# Loom kernels in Gufo, the Gufo way

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
`runner_cli.cpp`, which runs the server's backend in-process. Packaging is
this repository's `nix/runtime.nix`, a Nix package that builds hrx-system
v0.2.0 (the release loom-fun targets) with the embedding library, with the
nixpkgs Gufo pins; `withTools = true` adds loom-serve and loom-fetch for
side-by-side comparisons.

**Loom:** no source change. `libloom_text_embed.so` (about 1,000 lines of C
in the extension's `loom-embed/`) is compiled in Loom's CMake build (HRX
includes `loom-embed/loom_text_embed.cmake` once its targets exist) from the
existing text model layer:

- package loading via `model_configuration_load`/`resolve` and
  `text_model_allocate`, exactly as loom-serve registers a package;
- rows (`row_begin`, `row_prefill`, `row_decode`), packed `decode_batch`
  over `model_epoch`, drafted `verify_batch` over `model_verify`, and row
  checkpoints (`row_pin`/`row_restore`) for Gufo's prefix cache;
- checkpoint images (`checkpoint_export`/`checkpoint_import`): a checkpoint's
  pages, recurrent state and carry as position-independent bytes, using the
  same range plan and transfers Loom uses to suspend a checkpoint, for Gufo's
  disk tier;
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

## Logits

No Loom runtime change: `libloom_text_embed` is compiled inside Loom's tree and
already downloads device allocations (checkpoint images), so the package only
has to leave the logits somewhere the embed layer can read.

- **Package side** (loom-skills starter kit). A package declares `llm.logits`
  (card config, default false). With it true, the `outputs` allocation
  (storage slot 5) also holds one full f32 logit row per span from byte
  32768 on, row stride `ceil(vocab/128) * 512` bytes, written by the
  `@llm_advance` stage for spans whose request carries `"logits": true`.
  Default loads are unchanged by construction.
- **Embed ABI 2.** `loom_text_embed_open_ex(..., logits = 1)` rewrites the
  card's `llm.logits` to true in a private copy of the card (the package's
  programs and parameters stay where they are); the logit region is located
  from the allocation's size and checked. `"logits": true` in a request body
  marks the row bound with it; after every selecting epoch the rows of marked
  spans are downloaded in one transfer; `loom_text_embed_row_logits` returns
  the row of the pending selection; `loom_text_embed_decode_batch_tokens`
  runs an epoch on tokens the host chose; `loom_text_embed_is_stop_token`
  tells the stop token. ABI 1 callers are unchanged.
- **Runner.** A request needs the row when it carries a grammar this package
  can take, `logprobs`, `min_p` or penalties. Such a row runs undrafted:
  after the epoch that makes a selection the runner samples it from the row
  with Gufo's `SamplerState` (mask, penalties, min-p, seed), publishes that
  token and consumes it in the next epoch (`decode_batch_tokens`). The choice
  is made one epoch early (right after the row exists) so end of output and
  withheld tool calls are known with the last token, as on the device path.
  Whether the next epoch must leave a row is decided from the sampler as it
  will be once the token is accepted; a lazy tool grammar (`tool_choice`
  auto) stays on the device until it restricts. Grammars constrain the model's own tokens, not the canonical
  markup the runner publishes, so the runner describes the model's spelling to Gufo (`ToolGrammarSyntax` and
  `ReasoningGrammarMarkers`, part of Gufo's text runner contract since extension API 0.3.35): the package's
  `llm_config.json` (listed in `model.json`, so loom-fetch delivers it) names its tool-call format and reasoning markers and `src/loom_dialect.hpp` maps them
  to Gufo's `ToolSyntax` (Qwen XML, Hermes JSON, GLM tags, Gemma 4, OpenAI harmony, Llama JSON, Python-style call
  lists with keyword arguments, and Glimmer's ATEM messages; the last two use the descriptor's `block_close` /
  `call_separator`, `ValueSyntax::keyword_root` and `repeat_prefix`/`repeat_suffix`). Parsing of the published
  markup is unchanged.
- **Tests.** `tests/build_test.sh` builds the runner against a fake embed
  library (a byte-level model with scripted logits) and checks the device path
  is untouched, logprobs, penalties, min-p, seeds, JSON-schema and tool
  grammars, lazy tool grammars, every dialect's grammar against the call text the
  kit's chat templates render, dialect gating and mixed batches, without a GPU.

## Limits (and what removes them)

- **Features that need logits** (`tool_choice` required/named, strict
  schemas, `response_format` enforcement, logprobs, min_p, penalties) need a
  package built with the loom-skills kit's opt-in logits readback; see
  "Logits" below. Packages without it behave as before: they sample on device
  and those requests are served unconstrained, as in loom-serve.
- **Disk tier needs elastic state.** With `--cache-disk`, cache entries are
  portable checkpoint images, which need Loom's elastic pool: the extension
  uses it then (otherwise a fixed pool, as the loom-fun cards serve). A preset
  whose state still cannot be imaged keeps the cache in device memory (the
  runner says so at load).
- **Greedy outputs can differ from loom-serve at near-ties.** Loom's results
  depend on batch shape (drafted verify windows, prefill chunks); loom-serve
  itself gives a different greedy haiku for Qwen3-8B under load.
- **Containers need `--security-opt seccomp=unconfined`**, because Loom's GPU
  memory reservation fails under Docker's default profile.

## Evidence (Strix Halo gfx1151, 2026-11-14, Loom 0.1.0)

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
