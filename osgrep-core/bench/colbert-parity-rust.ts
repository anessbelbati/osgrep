/**
 * ColBERT Parity (Rust vs Rust)
 *
 * Compares:
 * - `colbertRerankPreindexed` (packed fast path)
 * - `colbertRerankOrt` (encode docs at query-time)
 *
 * This avoids Python/onnxruntime issues and validates that packed scoring + index
 * mapping are correct.
 *
 * Input: JSONL produced by `bench/colbert-parity.ts`
 *
 * Usage:
 *   source ~/.cargo/env
 *   bun run bench/colbert-parity-rust.ts --in /tmp/colbert-parity-chi.jsonl --topk 20
 */

import { readFileSync } from "fs";

// @ts-ignore - generated at build time
import { initColbertOrt, colbertPreindexDocs, colbertRerankOrt, colbertRerankPreindexed } from "../index.js";

function parseArgs(argv: string[]) {
  const args: Record<string, string> = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "1";
    args[key] = val;
  }
  return args;
}

type ParityMeta = {
  type: "meta";
  repo: string;
  colbertModelRepo: string;
  colbertHiddenSize: number;
  topk: number;
};

type ParityQuery = {
  type: "query";
  queryIndex: number;
  query: string;
  candidates: Array<{ chunkIndex: number; text: string }>;
  rust: { indices: number[]; scores: number[] };
};

function topKOverlap(a: number[], b: number[], k: number): number {
  const sa = new Set(a.slice(0, k));
  const sb = new Set(b.slice(0, k));
  let inter = 0;
  for (const x of sa) if (sb.has(x)) inter++;
  return inter / k;
}

async function main() {
  const args = parseArgs(process.argv);
  const inPath = args.in;
  const topk = Number(args.topk ?? 20);
  if (!inPath) throw new Error("Missing --in /path/to/parity.jsonl");

  const lines = readFileSync(inPath, "utf8").split("\n").filter(Boolean);
  const meta = JSON.parse(lines[0]) as ParityMeta;
  const queries = lines.slice(1).map(l => JSON.parse(l) as ParityQuery);

  initColbertOrt(meta.colbertModelRepo, meta.colbertHiddenSize);

  // Re-preindex docs in the same order as the export expects.
  // We can just union all candidate docs (order stable within each query) into a global
  // list and preindex them; but preindexed rerank expects global doc indices.
  // The export already used repo-level chunk indices, so for this check we instead
  // rebuild a *local* packed store per query: preindex the candidate docs only.
  //
  // That lets us compare packed-vs-nonpacked scoring without needing the full repo.
  let n = 0;
  let top1Match = 0;
  let top5OverlapSum = 0;

  for (const q of queries) {
    const docs = q.candidates.map(c => c.text);
    if (docs.length === 0) continue;

    colbertPreindexDocs(docs);

    // Packed path (indices will be local [0..docs.length))
    const packed = colbertRerankPreindexed(q.query, docs.map((_, i) => i), topk);
    const packedPos = Array.from(packed.indices) as number[];

    // Non-packed path (indices are local [0..docs.length))
    const nonPacked = colbertRerankOrt(q.query, docs, topk);
    const nonPackedPos = Array.from(nonPacked.indices) as number[];

    if (packedPos.length > 0 && nonPackedPos.length > 0 && packedPos[0] === nonPackedPos[0]) {
      top1Match++;
    }
    top5OverlapSum += topKOverlap(packedPos, nonPackedPos, Math.min(5, topk));
    n++;
  }

  if (n === 0) {
    console.log("No comparable queries");
    return;
  }

  console.log("============================================================");
  console.log("Rust vs Rust Parity (preindexed vs query-time)");
  console.log("============================================================");
  console.log(`Input: ${inPath}`);
  console.log(`Queries compared: ${n}`);
  console.log(`Top-1 match:   ${(top1Match / n * 100).toFixed(1)}%`);
  console.log(`Top-5 overlap: ${(top5OverlapSum / n).toFixed(3)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

