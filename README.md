# loom-gufo-extension

Runs [loom-fun](https://huggingface.co/loom-fun) Loom model packages in
[Gufo](https://github.com/gufo-org/gufo) on Strix Halo. Each package is
served by `loom-serve` behind Gufo's OpenAI, Responses and Anthropic APIs.

Needs Gufo with request-level text services (extension API >= 0.3.15) and
`loom-serve` from `hrx-loom` 0.1.0 (`pip install hrx-loom==0.1.0`).

```sh
loom-fetch huggingface:loom-fun/ggml-org-Qwen3-4B-loom --output=models/qwen3-4b
gufo serve llm --model models/qwen3-4b
```

Options: `--loom-serve PATH`, `--loom-device URI`, `--loom-memory-bytes N`,
`--loom-preset NAME`.

In a container, add `--security-opt seccomp=unconfined`: Loom's GPU memory
reservation fails under Docker's default seccomp profile.
