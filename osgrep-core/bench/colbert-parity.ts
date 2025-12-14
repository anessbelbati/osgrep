/**
 * ColBERT Parity Export
 *
 * Generates an apples-to-apples dataset to compare:
 * - Rust/ORT implementation (preindexed rerank)
 * - Python ONNX MaxSim scorer (should match Rust ranking)
 *
 * Output JSONL contains per-query candidate docs + Rust top-k.
 *
 * Usage:
 *   source ~/.cargo/env
 *   bun run bench/colbert-parity.ts --repo chi --n 30 --lines 20 --dense-topk 100 --topk 20 --out /tmp/parity.jsonl
 */
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import { loadRepoChunksWithMeta } from "./_util.js";

// @ts-ignore - generated at build time
import {
  initDenseOrt,
  denseEmbedOrt,
  initColbertOrt,
  colbertPreindexDocs,
  colbertRerankPreindexed,
} from "../index.js";

type PositiveSpan = { file: string; start_line: number; end_line: number };

type CodeAtlasRow = {
  repo: string;
  query: string;
  positives: PositiveSpan[];
  difficulty?: string;
  category?: string;
};

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

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-12);
}

function denseTopK(query: string, docEmbeddings: number[][], topK: number): number[] {
  const queryResult = denseEmbedOrt([query], true);
  const queryEmb = Array.from(queryResult.embeddings) as number[];

  const scored: Array<{ idx: number; score: number }> = [];
  for (let i = 0; i < docEmbeddings.length; i++) {
    scored.push({ idx: i, score: cosineSimilarity(queryEmb, docEmbeddings[i]) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(s => s.idx);
}

async function main() {
  const args = parseArgs(process.argv);

  const repo = args.repo ?? "chi";
  const n = Number(args.n ?? 30);
  const linesPerChunk = Number(args.lines ?? 20);
  const denseTopk = Number(args["dense-topk"] ?? 100);
  const topk = Number(args.topk ?? 20);
  const outPath = args.out ?? `/tmp/colbert-parity-${repo}.jsonl`;

  const codeatlasPath = args.codeatlas ?? "./codeAtlas/artifacts/codeatlas.jsonl";
  const reposDir = args.repos ?? "./codeAtlas/repos_to_test";

  const denseModelRepo = args["dense-model"] ?? "onnx-community/granite-embedding-30m-english-ONNX";
  const denseHiddenSize = Number(args["dense-dim"] ?? 384);
  const colbertModelRepo = args["colbert-model"] ?? "ryandono/osgrep-17m-v1-onnx";
  const colbertHiddenSize = Number(args["colbert-dim"] ?? 48);

  if (!existsSync(codeatlasPath)) {
    throw new Error(`Missing CodeAtlas JSONL at ${codeatlasPath}`);
  }

  const repoDir = join(reposDir, repo);
  if (!existsSync(repoDir)) {
    throw new Error(`Missing repo dir at ${repoDir}`);
  }

  const rows: CodeAtlasRow[] = [];
  for (const line of require("fs").readFileSync(codeatlasPath, "utf8").split("\n")) {
    if (!line) continue;
    const r = JSON.parse(line) as CodeAtlasRow;
    if (r.repo === repo) rows.push(r);
  }
  const picked = rows.slice(0, n);
  if (picked.length === 0) throw new Error(`No rows for repo=${repo}`);

  console.log(`[parity] repo=${repo} queries=${picked.length} linesPerChunk=${linesPerChunk}`);

  console.log(`[parity] init dense=${denseModelRepo}`);
  initDenseOrt(denseModelRepo, denseHiddenSize);
  console.log(`[parity] init colbert=${colbertModelRepo}`);
  initColbertOrt(colbertModelRepo, colbertHiddenSize);

  console.log(`[parity] chunking...`);
  const chunks = loadRepoChunksWithMeta(repoDir, linesPerChunk, repo);
  const chunkTexts = chunks.map(c => c.text);

  console.log(`[parity] dense embedding docs=${chunkTexts.length}...`);
  const batchSize = 64;
  const allDocEmbeddings: number[][] = [];
  for (let i = 0; i < chunkTexts.length; i += batchSize) {
    const batch = chunkTexts.slice(i, i + batchSize);
    const res = denseEmbedOrt(batch, true);
    const flat = Array.from(res.embeddings) as number[];
    const hs = res.hiddenSize as number;
    for (let j = 0; j < batch.length; j++) {
      allDocEmbeddings.push(flat.slice(j * hs, (j + 1) * hs));
    }
  }

  console.log(`[parity] preindex colbert docs=${chunkTexts.length}...`);
  colbertPreindexDocs(chunkTexts);

  const outLines: string[] = [];
  outLines.push(JSON.stringify({
    type: "meta",
    repo,
    out: outPath,
    denseModelRepo,
    denseHiddenSize,
    colbertModelRepo,
    colbertHiddenSize,
    linesPerChunk,
    denseTopk,
    topk,
    totalChunks: chunks.length,
  }));

  for (let i = 0; i < picked.length; i++) {
    const row = picked[i];
    const candIdx = denseTopK(row.query, allDocEmbeddings, Math.min(denseTopk, chunks.length));
    const rust = colbertRerankPreindexed(row.query, candIdx, topk);

    const candidates = candIdx.map(idx => ({
      chunkIndex: idx,
      file: chunks[idx]?.file ?? "",
      startLine: chunks[idx]?.startLine ?? 0,
      endLine: chunks[idx]?.endLine ?? 0,
      text: chunkTexts[idx] ?? "",
    }));

    outLines.push(JSON.stringify({
      type: "query",
      repo,
      queryIndex: i,
      query: row.query,
      positives: row.positives,
      category: row.category,
      difficulty: row.difficulty,
      candidates,
      rust: {
        indices: Array.from(rust.indices as number[]),
        scores: Array.from(rust.scores as number[]),
      },
    }));
  }

  writeFileSync(outPath, outLines.join("\n") + "\n", "utf8");
  console.log(`[parity] wrote ${outLines.length} lines -> ${outPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

