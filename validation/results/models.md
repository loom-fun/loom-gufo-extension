# Declared models through Gufo

Each model as a user runs it: `gufo model install ID`, then `gufo serve llm --model ID --sessions 4`. Strix Halo test.

Gufo checks: `conformance.py --logits` (29 checks). Throughput: `bench.py --reps 2` (greedy, thinking off;
decode = one stream after a short prompt, prefill = time to first token of a ~10k-token document, 4 streams =
aggregate decode). Gufo / loom-serve: the same package, preset and flags under loom-serve 0.2.0 on the same machine.

| Model ID | Gufo checks | Decode tok/s | Prefill tok/s | 4 streams tok/s | Gufo / loom-serve decode, prefill, 4 streams | Notes |
|---|---|---|---|---|---|---|
| `qwen3-0.6b-loom` | 29/29 | 344.4 | 8,674 | 1,141.2 |  |  |
| `qwen2.5-0.5b-instruct-loom` | 29/29 | 366.9 | 14,099 | 1,292.4 |  |  |
| `qwen3.5-0.8b-loom` | 29/29 | 258.5 | 11,731 | 764.2 |  |  |
| `gemma-3-1b-it-loom` | 21/28 | 207.7 | 11,190 | 663.9 |  | no tool calls (model) |
| `qwen2.5-1.5b-instruct-loom` | 29/29 | 182.5 | 5,894 | 639.5 |  |  |
| `lfm3-omni-2b-loom` | 29/29 | 172.3 | 7,329 | 614.5 |  |  |
| `lfm3-1.7b-loom` | 29/29 | 172.0 | 7,170 | 608.6 |  |  |
| `hyperclovax-seed-text-instruct-1.5b-loom` | 21/28 | 181.1 | 5,969 | 642.3 |  | no tool calls (model) |
| `qwen3-1.7b-loom` | 29/29 | 164.7 | 5,153 | 546.0 |  |  |
| `qwen3.5-2b-loom` | 28/29 | 145.0 | 6,442 | 383.5 |  | reasoning effort: low reasoning 8181 chars, high 8181 chars |
| `llama-3.2-3b-instruct-loom` | 28/29 | 98.5 | 2,901 | 309.2 |  | tool round trip:  |
| `qwen3-4b-instruct-2507-loom` | 29/29 | 77.4 | 2,255 | 250.1 |  |  |
| `spark-x2.5-4b-loom` | 29/29 | 72.7 | 2,774 | 254.0 |  |  |
| `spark-x2.6-4b-loom` | 29/29 | 72.5 | 2,753 | 218.9 |  |  |
| `qwen3-4b-loom` | 29/29 | 104.8 | 2,209 | 225.9 |  |  |
| `qwen3-vl-4b-instruct-loom` | 29/29 | 78.3 | 2,241 | 266.1 |  |  |
| `qwen3.5-4b-loom` | 29/29 | 90.9 | 2,930 | 192.9 |  |  |
| `gemma-4-e2b-it-loom` | 29/29 | 151.8 | 3,834 | 396.6 |  |  |
| `qwen2.5-7b-instruct-loom` | 29/29 | 48.2 | 1,879 | 163.4 |  |  |
| `qwen2.5-vl-7b-instruct-loom` | 21/28 | 47.8 | 1,864 | 168.8 |  | no tool calls (model) |
| `gemma-4-e4b-it-loom` | 29/29 | 85.9 | 2,528 | 219.1 |  |  |
| `qwen3-8b-loom` | not run | | | | | |
| `qwen3-vl-8b-instruct-loom` | not run | | | | | |
| `ornith-1.0-9b-loom` | not run | | | | | |
| `gemma-4-12b-it-loom` | not run | | | | | |
| `ornith-1.5-9b-loom` | not run | | | | | |
| `nemotron-3.5-bolt-e4b-loom` | not run | | | | | |
| `qwen3.5-9b-loom` | not run | | | | | |
| `mellum2.1-12b-a2.5b-thinking-loom` | not run | | | | | |
| `qwen3-14b-loom` | not run | | | | | |
| `gpt-oss-20b-loom` | not run | | | | | |
| `lfm2.5-24b-a2b-loom` | not run | | | | | |
| `gemma-4-26b-a4b-it-loom` | not run | | | | | |
| `qwen3-coder-30b-a3b-instruct-loom` | not run | | | | | |
| `qwen3.8-27b-loom` | not run | | | | | |
| `qwen3-30b-a3b-instruct-2507-loom` | not run | | | | | |
| `gemma-4-31b-it-loom` | not run | | | | | |
| `qwen2.5-32b-instruct-loom` | not run | | | | | |
| `laguna-xs-2.1-loom` | not run | | | | | |
| `nanbeige4.3-33b-loom` | not run | | | | | |
| `qwen3.6-27b-loom` | not run | | | | | |
| `qwen3.6-35b-a3b-loom` | not run | | | | | |
| `ornith-1.0-35b-loom` | not run | | | | | |
| `minicpm-v-4.7-35b-a3b-loom` | not run | | | | | |
| `sage-a1.5-loom` | not run | | | | | |
| `qwen3.5-35b-a3b-loom` | not run | | | | | |
| `ornith-1.5-35b-a3b-loom` | not run | | | | | |
| `muse-glimmer-30b-loom` | not run | | | | | |
