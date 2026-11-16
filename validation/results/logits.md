# Logits: Gufo's logit-dependent features on loom-fun packages

Run 2026-11-15 on Strix Halo (gfx1151): the 48 loom-fun-staging packages of 35B parameters or less, fetched with
`loom-fetch` as users get them, served with `--loom-logits=on`. `conformance.py --logits` (29 checks: the usual Gufo
features plus `tool_choice` required/named, strict tool schemas, `response_format` json_schema, min_p/penalties and
logprobs, all of which must pass, not be gaps) and `devlab/logits_check.py` (host greedy == device greedy, grammar
and plain requests batched together).

42 of 48 pass every check; the others fail only on package or model behavior.

| Model | conformance | logits_check | Not passing |
|---|---|---|---|
| Qwen/Qwen3-0.6B | 29/29 | 2/2 |  |
| Qwen/Qwen2.5-0.5B-Instruct | 29/29 | 2/2 |  |
| Qwen/Qwen3.5-0.8B | 29/29 | 2/2 |  |
| google/gemma-3-1b-it | 21/28 | 2/2 | tool call (auto), streamed tool call, tool_choice required, tool_choice named, strict tool schema, Responses API (text + function call), Anthropic Messages (text + tool_use) (package has no tool calls) |
| Qwen/Qwen2.5-1.5B-Instruct | 29/29 | 2/2 |  |
| LiquidAI/LFM3-Omni-2B | 29/29 | 2/2 |  |
| LiquidAI/LFM3-1.7B | 29/29 | 2/2 |  |
| naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B | 21/28 | 2/2 | tool call (auto), streamed tool call, tool_choice required, tool_choice named, strict tool schema, Responses API (text + function call), Anthropic Messages (text + tool_use) (package has no tool calls) |
| Qwen/Qwen3-1.7B | 29/29 | 2/2 |  |
| Qwen/Qwen3.5-2B | 28/29 | 2/2 | reasoning effort (reasoning loops) |
| meta-llama/Llama-3.2-3B-Instruct | 28/29 | 2/2 | tool round trip (model calls again instead of answering (loom-serve too)) |
| Qwen/Qwen3-4B-Instruct-2507 | 29/29 | 2/2 |  |
| XHToken/Spark-X2.5-4B | 29/29 | 2/2 |  |
| XHToken/Spark-X2.6-4B | 29/29 | 2/2 |  |
| Qwen/Qwen3-4B | 29/29 | 2/2 |  |
| Qwen/Qwen3-VL-4B-Instruct | 29/29 | 2/2 |  |
| Qwen/Qwen3.5-4B | 29/29 | 2/2 |  |
| google/gemma-4-E2B-it | 29/29 | 2/2 |  |
| Qwen/Qwen2.5-7B-Instruct | 29/29 | 2/2 |  |
| Qwen/Qwen2.5-VL-7B-Instruct | 21/28 | 2/2 | tool call (auto), streamed tool call, tool_choice required, tool_choice named, strict tool schema, Responses API (text + function call), Anthropic Messages (text + tool_use) (package has no tool calls) |
| google/gemma-4-E4B-it | 29/29 | 2/2 |  |
| Qwen/Qwen3-8B | 29/29 | 2/2 |  |
| Qwen/Qwen3-VL-8B-Instruct | 29/29 | 2/2 |  |
| deepreinforce-ai/Ornith-1.0-9B | 29/29 | 2/2 |  |
| google/gemma-4-12B-it | 29/29 | 2/2 |  |
| ornith-ai/Ornith-1.5-9B | 29/29 | 2/2 |  |
| nemotron-3.5-bolt-e4b | 29/29 | 2/2 |  |
| Qwen/Qwen3.5-9B | 29/29 | 2/2 |  |
| JetBrains/Mellum2.1-12B-A2.5B-Thinking | 29/29 | 2/2 |  |
| Qwen/Qwen3-14B | 29/29 | 2/2 |  |
| openai/gpt-oss-20b | 29/29 | 2/2 |  |
| LiquidAI/LFM2.5-24B-A2B | 29/29 | 2/2 |  |
| google/gemma-4-26B-A4B-it | 29/29 | 2/2 |  |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 29/29 | 2/2 |  |
| qwen3.8-27b | 29/29 | 2/2 |  |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | 29/29 | 2/2 |  |
| google/gemma-4-31B-it | 29/29 | 2/2 |  |
| Qwen/Qwen2.5-32B-Instruct | 29/29 | 2/2 |  |
| poolside/Laguna-XS-2.1 | 29/29 | 2/2 |  |
| Nanbeige/Nanbeige4.3-33B | 29/29 | 2/2 |  |
| Qwen/Qwen3.6-27B | 29/29 | 2/2 |  |
| Qwen/Qwen3.6-35B-A3B | 29/29 | 2/2 |  |
| ornith-ai/Ornith-1.0-35B | 28/29 | 2/2 | tool_choice named (reasoning loops when the named tool conflicts with the question) |
| openbmb/MiniCPM-V-4.7-35B-A3B | 29/29 | 2/2 |  |
| InternScience/Sage-A1.5 | 29/29 | 2/2 |  |
| Qwen/Qwen3.5-35B-A3B | 29/29 | 2/2 |  |
| ornith-ai/Ornith-1.5-35B-A3B | 29/29 | 2/2 |  |
| meta-models/Muse-Glimmer-30B | 29/29 | 2/2 |  |
