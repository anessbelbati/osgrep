# ColBERT Parity (Rust ORT vs Python ONNX)

This repo has a small “apples-to-apples” parity harness to check whether the
Rust ORT ColBERT reranker matches a reference Python ONNX MaxSim implementation
on identical queries and identical candidate chunks.

## 1) Export parity input (Node/Rust)

From the repo root:

```bash
source ~/.cargo/env
bun run bench/colbert-parity.ts --repo chi --n 30 --lines 20 --dense-topk 100 --topk 20 --out /tmp/colbert-parity-chi.jsonl
```

This writes JSONL with:
- the query text
- the dense candidate set (chunk texts + metadata)
- the Rust reranker’s returned indices/scores for the same candidate set

## 2) Run Python ONNX scorer on the exported JSONL

You need paths to:
- `model_int8.onnx` (or `model.onnx`)
- `tokenizer.json`
- optional `skiplist.json`

If you’re using the same HF Hub model as Rust (`ryandono/osgrep-17m-v1-onnx`),
these should exist in your local HuggingFace cache under:

`~/.cache/huggingface/hub/models--ryandono--osgrep-17m-v1-onnx/snapshots/<SNAPSHOT>/`

Run:

```bash
python3 codeAtlas/python/parity_rust_vs_onnx.py \
  --in /tmp/colbert-parity-chi.jsonl \
  --onnx-model ~/.cache/huggingface/hub/models--ryandono--osgrep-17m-v1-onnx/snapshots/<SNAPSHOT>/onnx/model_int8.onnx \
  --tokenizer-json ~/.cache/huggingface/hub/models--ryandono--osgrep-17m-v1-onnx/snapshots/<SNAPSHOT>/tokenizer.json \
  --skiplist-json ~/.cache/huggingface/hub/models--ryandono--osgrep-17m-v1-onnx/snapshots/<SNAPSHOT>/skiplist.json \
  --topk 20
```

Expected: very high Top-1 match and Top-5 overlap if preprocessing is aligned.

