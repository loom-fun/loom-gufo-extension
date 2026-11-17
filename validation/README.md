# Validation

Clients that check a Gufo server running Loom packages (or loom-serve itself,
for comparison) through its public APIs. Each takes the server's base URL.

| Script | What it checks |
|---|---|
| `conformance.py BASE OUT.json` | 29 Gufo features: endpoints, chat, streaming, reasoning on/off, budgets, token usage and effort, tool calls (Chat, Responses, Messages), stop/finish, sampling, raw completions, tokenization, prefix cache (reuse, exact restore, shared prefix), concurrency, cancellation; the logits-dependent ones (tool_choice required and named, strict tool schemas, response_format, min_p and penalties, logprobs) are gaps without the package capability and must pass with `--logits` |
| `vision_test.py BASE MODEL` | image input (color recognition), image cache identity, Messages image parts |
| `ref.py BASE MODEL OUT.json` | greedy reference answers and behavior probes, for parity with loom-serve |
| `parity.py BASE` | tool-call and reasoning behavior, to tell model behavior from integration bugs |
| `bench.py BASE MODEL OUT.json --doc PATH` | decode, prefill and 4-stream throughput, TTFT |
| `disk_check.py BASE OUT.json first\|second` | disk tier: run on a server with `--cache-disk`, restart it, run `second`; passes when the restarted server reuses the prompt from disk with the same greedy answer (`disk_test.py` is the request) |
| `thinkoff.py`, `haikuload.py` | thinking-off handling; greedy variance of loom-serve under load |

Results in `results/`: `conformance.md` (Loom 0.2.0, 2026-11-17, and the full 0.1.0 run of 2026-11-14), `bench.md` and
`logits.md` (Loom 0.1.0, Strix Halo gfx1151).
