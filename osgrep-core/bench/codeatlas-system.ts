/**
 * CodeAtlas System Benchmark
 *
 * Tests the full osgrep pipeline:
 * 1. Chunk repo with actual chunker (80 lines)
 * 2. Dense retrieval (ORT granite-30m) → top 100 candidates
 * 3. ColBERT rerank (pre-indexed) → top 5 results
 * 4. Score hits by checking chunk/span overlap
 */

import { readFileSync, existsSync } from "fs";
import { join, extname } from "path";
import {
  loadRepoChunksWithMeta,
  ChunkMeta,
  chunkOverlapsSpan,
} from "./_util.js";

// @ts-ignore - generated at build time
import {
  initDenseOrt,
  denseEmbedOrt,
  initColbertOrt,
  colbertPreindexDocs,
  colbertRerankPreindexed,
} from "../index.js";

// =============================================================================
// Configuration
// =============================================================================

const CODEATLAS_PATH = "./codeAtlas/artifacts/codeatlas.jsonl";
const REPOS_DIR = "./codeAtlas/repos_to_test";

// Models
const DENSE_MODEL_REPO = "onnx-community/granite-embedding-30m-english-ONNX";
const DENSE_HIDDEN_SIZE = 384;
const COLBERT_MODEL_REPO = "ryandono/osgrep-17m-v1-onnx";
const COLBERT_HIDDEN_SIZE = 48;

// Retrieval params
const DENSE_TOP_K = 100;
const COLBERT_TOP_K = 5;
const LINES_PER_CHUNK = Number(process.env.LINES_PER_CHUNK ?? 80);
const EXCLUDE_NON_CODE = (process.env.EXCLUDE_NON_CODE ?? "1") !== "0";
const INCLUDE_PATH_HEADER = (process.env.INCLUDE_PATH_HEADER ?? "0") !== "0";
const EVAL_DENSE_ONLY = (process.env.EVAL_DENSE_ONLY ?? "1") !== "0";
const EVAL_COLBERT_ONLY = (process.env.EVAL_COLBERT_ONLY ?? "0") !== "0";
const HYBRID_ALPHA = process.env.HYBRID_ALPHA ? Number(process.env.HYBRID_ALPHA) : null;

// Repos to test (start with smaller repos for quick validation)
const TEST_REPOS = (() => {
  const env = (process.env.TEST_REPOS ?? "").trim();
  if (!env) {
    return new Set([
      "chi",
    ]);
  }
  return new Set(env.split(",").map(s => s.trim()).filter(Boolean));
})();

// =============================================================================
// Types
// =============================================================================

interface CodeAtlasRow {
  repo: string;
  query: string;
  positives: Array<{ file: string; start_line: number; end_line: number; snippet?: string }>;
  negatives?: Array<{ file: string; start_line: number; end_line: number; snippet?: string }>;
  category?: string;
  difficulty?: string;
}

interface QueryResult {
  query: string;
  repo: string;
  category?: string;
  difficulty?: string;
  denseRecallAt100: boolean;
  denseHitAt5: boolean;
  denseMrrAt5: number;
  colbertOnlyHitAt5?: boolean;
  colbertOnlyMrrAt5?: number;
  hybridHitAt5?: boolean;
  hybridMrrAt5?: number;
  hitAt1: boolean;
  hitAt5: boolean;
  mrrAt5: number;
  bestRankAt5: number;
  numPositives: number;
  numChunks: number;
  timeMs: number;
}

// =============================================================================
// Dense Retrieval
// =============================================================================

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

function denseRetrieveScored(
  queryText: string,
  docEmbeddings: number[][],
  topK: number
): Array<{ idx: number; score: number }> {
  // Encode query
  const queryResult = denseEmbedOrt([queryText], true);
  const queryEmb = Array.from(queryResult.embeddings);

  // Compute similarities
  const scores: Array<{ idx: number; score: number }> = [];
  for (let i = 0; i < docEmbeddings.length; i++) {
    const score = cosineSimilarity(queryEmb, docEmbeddings[i]);
    scores.push({ idx: i, score });
  }

  // Sort descending and take top K
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topK);
}

function minMaxNormalize(values: number[]): number[] {
  if (values.length === 0) return values;
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  if (range <= 1e-12) return values.map(() => 0.5);
  return values.map(v => (v - min) / range);
}

// =============================================================================
// Metrics
// =============================================================================

function computeMetricsAtK(
  rankedIndices: number[],
  chunks: ChunkMeta[],
  positives: CodeAtlasRow["positives"]
): { hitAt1: boolean; hitAt5: boolean; mrr: number; bestRank: number } {
  let bestRank = Infinity;

  for (let rank = 0; rank < rankedIndices.length; rank++) {
    const chunkIdx = rankedIndices[rank];
    const chunk = chunks[chunkIdx];

    // Check if this chunk overlaps any positive span
    const isHit = positives.some(pos => chunkOverlapsSpan(chunk, pos));
    if (isHit && rank + 1 < bestRank) {
      bestRank = rank + 1;  // 1-indexed rank
    }
  }

  if (bestRank === Infinity) {
    return { hitAt1: false, hitAt5: false, mrr: 0, bestRank: -1 };
  }

  return {
    hitAt1: bestRank === 1,
    hitAt5: bestRank <= 5,
    mrr: 1 / bestRank,
    bestRank,
  };
}

function positiveChunkIndices(chunks: ChunkMeta[], positives: CodeAtlasRow["positives"]): Set<number> {
  const indices = new Set<number>();
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    if (positives.some(pos => chunkOverlapsSpan(chunk, pos))) {
      indices.add(i);
    }
  }
  return indices;
}

function isLikelyCodeFile(filePath: string): boolean {
  const ext = extname(filePath).toLowerCase();
  // Allowlist focused on code; excludes docs/JSON noise that often dominates rerank.
  const allow = new Set([
    ".py", ".pyx",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".rs", ".go",
    ".java", ".kt", ".scala",
    ".cs",
    ".rb", ".php", ".ex", ".exs",
    ".c", ".cc", ".cpp", ".h", ".hpp",
    ".swift",
    ".sql", ".proto",
  ]);
  return allow.has(ext);
}

// =============================================================================
// Main Benchmark
// =============================================================================

async function main() {
  console.log("=== CodeAtlas System Benchmark ===\n");

  // 1. Load CodeAtlas data
  console.log("Loading CodeAtlas data...");
  if (!existsSync(CODEATLAS_PATH)) {
    console.error(`CodeAtlas file not found: ${CODEATLAS_PATH}`);
    process.exit(1);
  }

  const lines = readFileSync(CODEATLAS_PATH, "utf-8").split("\n").filter(Boolean);
  const allRows: CodeAtlasRow[] = lines.map(l => JSON.parse(l));

  // Filter to test repos
  const rows = allRows.filter(r => TEST_REPOS.has(r.repo));
  console.log(`  Total rows: ${allRows.length}`);
  console.log(`  Test repos: ${Array.from(TEST_REPOS).join(", ")}`);
  console.log(`  Filtered rows: ${rows.length}\n`);

  // 2. Group by repo
  const byRepo = new Map<string, CodeAtlasRow[]>();
  for (const row of rows) {
    if (!byRepo.has(row.repo)) byRepo.set(row.repo, []);
    byRepo.get(row.repo)!.push(row);
  }

  // 3. Initialize models
  console.log("Initializing models...");
  const initStart = performance.now();
  initDenseOrt(DENSE_MODEL_REPO, DENSE_HIDDEN_SIZE);
  initColbertOrt(COLBERT_MODEL_REPO, COLBERT_HIDDEN_SIZE);
  const initTime = performance.now() - initStart;
  console.log(`  Models loaded in ${initTime.toFixed(0)}ms\n`);

  // 4. Process each repo
  const allResults: QueryResult[] = [];
  let totalIndexTime = 0;
  let totalQueryTime = 0;

  for (const [repo, repoRows] of byRepo) {
    const repoDir = join(REPOS_DIR, repo);
    if (!existsSync(repoDir)) {
      console.log(`⚠ Skipping ${repo} - repo not found at ${repoDir}`);
      continue;
    }

    console.log(`\n--- Processing ${repo} (${repoRows.length} queries) ---`);

    // 4a. Chunk the repo
    console.log(`  Chunking...`);
    const chunkStart = performance.now();
    // Pass repo name to match CodeAtlas file path format (e.g., "aiohttp/web_protocol.py")
    let chunks = loadRepoChunksWithMeta(repoDir, LINES_PER_CHUNK, repo);
    if (EXCLUDE_NON_CODE) {
      chunks = chunks.filter(c => isLikelyCodeFile(c.file));
    }
    const chunkTime = performance.now() - chunkStart;
    console.log(`  ${chunks.length} chunks in ${chunkTime.toFixed(0)}ms`);

    if (chunks.length === 0) {
      console.log(`  ⚠ No chunks, skipping`);
      continue;
    }

    // 4b. Dense embed all chunks
    console.log(`  Dense indexing...`);
    const denseStart = performance.now();
    const chunkTexts = chunks.map(c => {
      if (!INCLUDE_PATH_HEADER) return c.text;
      return `FILE: ${c.file}\nLINES: ${c.startLine}-${c.endLine}\n${c.text}`;
    });

    // Batch encode in groups of 64 to avoid memory issues
    const batchSize = 64;
    const allDocEmbeddings: number[][] = [];

    for (let i = 0; i < chunkTexts.length; i += batchSize) {
      const batch = chunkTexts.slice(i, i + batchSize);
      const result = denseEmbedOrt(batch, true);
      const embeddings = Array.from(result.embeddings);
      const hs = result.hiddenSize;

      for (let j = 0; j < batch.length; j++) {
        const start = j * hs;
        const end = start + hs;
        allDocEmbeddings.push(embeddings.slice(start, end));
      }
    }

    const denseTime = performance.now() - denseStart;
    console.log(`  Dense indexed in ${denseTime.toFixed(0)}ms`);

    // 4c. ColBERT preindex all chunks
    console.log(`  ColBERT preindexing...`);
    const colbertStart = performance.now();
    colbertPreindexDocs(chunkTexts);
    const colbertTime = performance.now() - colbertStart;
    console.log(`  ColBERT preindexed in ${colbertTime.toFixed(0)}ms`);

    totalIndexTime += denseTime + colbertTime;

    // 4d. Run queries
    console.log(`  Running queries...`);
    for (const row of repoRows) {
      const queryStart = performance.now();
      const posSet = positiveChunkIndices(chunks, row.positives);

      // Dense retrieval → top 100
      const denseTopKScored = denseRetrieveScored(
        row.query,
        allDocEmbeddings,
        DENSE_TOP_K
      );
      const denseTopK = denseTopKScored.map(s => s.idx);
      const denseRecallAt100 = denseTopK.some(i => posSet.has(i));
      const denseTop5 = denseTopK.slice(0, Math.min(5, denseTopK.length));
      const denseMetricsAt5 = EVAL_DENSE_ONLY ? computeMetricsAtK(denseTop5, chunks, row.positives) : { hitAt1: false, hitAt5: false, mrr: 0, bestRank: -1 };

      // ColBERT rerank → top 5
      // colbertRerankPreindexed returns indices into the candidate set (denseTopK)
      // and those are already mapped back to original doc indices by the Rust code
      const colbertResult = colbertRerankPreindexed(
        row.query,
        denseTopK,  // Pass the candidate indices
        HYBRID_ALPHA !== null ? denseTopK.length : COLBERT_TOP_K
      );

      // colbertResult.indices contains the original chunk indices (not indices into denseTopK)
      const finalRanking = Array.from(colbertResult.indices) as number[];

      const queryTime = performance.now() - queryStart;
      totalQueryTime += queryTime;

      // Validate indices before computing metrics
      const validRanking = finalRanking.filter(idx => idx >= 0 && idx < chunks.length);
      if (validRanking.length === 0) {
        console.log(`  ⚠ No valid rankings for query: ${row.query.slice(0, 50)}...`);
        continue;
      }

      // Score
      const metricsAt5 = computeMetricsAtK(validRanking.slice(0, 5), chunks, row.positives);

      // Optional hybrid: combine dense + colbert within denseTopK to stabilize reranking.
      // Hybrid ranking computed as: alpha * norm(colbert) + (1-alpha) * norm(dense)
      let hybridMetricsAt5: { hitAt1: boolean; hitAt5: boolean; mrr: number; bestRank: number } | null = null;
      if (HYBRID_ALPHA !== null && !Number.isNaN(HYBRID_ALPHA) && HYBRID_ALPHA >= 0 && HYBRID_ALPHA <= 1) {
        const alpha = HYBRID_ALPHA;
        const denseScores = denseTopKScored.map(s => s.score);
        const denseNorm = minMaxNormalize(denseScores);

        const colbertIdx = Array.from(colbertResult.indices) as number[];
        const colbertScores = Array.from(colbertResult.scores) as number[];
        const colbertScoreByChunk = new Map<number, number>();
        for (let i = 0; i < colbertIdx.length; i++) colbertScoreByChunk.set(colbertIdx[i], colbertScores[i]);

        const colbertScoresAligned = denseTopK.map(idx => colbertScoreByChunk.get(idx) ?? 0);
        const colbertNorm = minMaxNormalize(colbertScoresAligned);

        const hybridScored = denseTopK.map((idx, i) => ({
          idx,
          score: alpha * colbertNorm[i] + (1 - alpha) * denseNorm[i],
        }));
        hybridScored.sort((a, b) => b.score - a.score);
        const hybridTop5 = hybridScored.slice(0, 5).map(s => s.idx);
        hybridMetricsAt5 = computeMetricsAtK(hybridTop5, chunks, row.positives);
      }

      let colbertOnlyMetricsAt5: { hitAt1: boolean; hitAt5: boolean; mrr: number; bestRank: number } | null = null;
      if (EVAL_COLBERT_ONLY) {
        const allIdx = Array.from({ length: chunks.length }, (_, i) => i);
        const allRes = colbertRerankPreindexed(row.query, allIdx, 5);
        const allRank = (Array.from(allRes.indices) as number[]).filter(idx => idx >= 0 && idx < chunks.length);
        colbertOnlyMetricsAt5 = computeMetricsAtK(allRank, chunks, row.positives);
      }

      // Debug first few queries
      if (allResults.length < 3) {
        console.log(`\n  DEBUG Query: "${row.query.slice(0, 50)}..."`);
        console.log(`    Positives: ${row.positives.map(p => `${p.file}:${p.start_line}-${p.end_line}`).join(", ")}`);
        console.log(`    #positive chunks: ${posSet.size}  dense@100 hit: ${denseRecallAt100}`);
        console.log(`    Top 5 chunks: ${validRanking.slice(0, 5).map(i => `${chunks[i].file}:${chunks[i].startLine}-${chunks[i].endLine}`).join(", ")}`);
        console.log(`    Best rank@5: ${metricsAt5.bestRank}, Hit@1: ${metricsAt5.hitAt1}, Hit@5: ${metricsAt5.hitAt5}`);
      }

      allResults.push({
        query: row.query,
        repo,
        category: row.category,
        difficulty: row.difficulty,
        denseRecallAt100,
        denseHitAt5: denseMetricsAt5.hitAt5,
        denseMrrAt5: denseMetricsAt5.mrr,
        colbertOnlyHitAt5: colbertOnlyMetricsAt5?.hitAt5,
        colbertOnlyMrrAt5: colbertOnlyMetricsAt5?.mrr,
        hybridHitAt5: hybridMetricsAt5?.hitAt5,
        hybridMrrAt5: hybridMetricsAt5?.mrr,
        hitAt1: metricsAt5.hitAt1,
        hitAt5: metricsAt5.hitAt5,
        mrrAt5: metricsAt5.mrr,
        bestRankAt5: metricsAt5.bestRank,
        numPositives: row.positives.length,
        numChunks: chunks.length,
        timeMs: queryTime,
      });
    }
  }

  // =============================================================================
  // Report Results
  // =============================================================================

  console.log("\n" + "=".repeat(60));
  console.log("=== RESULTS ===");
  console.log("=".repeat(60) + "\n");

  const n = allResults.length;
  if (n === 0) {
    console.log("No results to report");
    return;
  }

  // Overall metrics
  const hitAt1 = allResults.filter(r => r.hitAt1).length / n;
  const hitAt5 = allResults.filter(r => r.hitAt5).length / n;
  const denseRecallAt100 = allResults.filter(r => r.denseRecallAt100).length / n;
  const denseHitAt5 = allResults.filter(r => r.denseHitAt5).length / n;
  const denseMrrAt5 = allResults.reduce((sum, r) => sum + r.denseMrrAt5, 0) / n;
  const mrrAt5 = allResults.reduce((sum, r) => sum + r.mrrAt5, 0) / n;
  const hybridHitAt5 = allResults.filter(r => r.hybridHitAt5).length / n;
  const hybridMrrAt5 = allResults.reduce((sum, r) => sum + (r.hybridMrrAt5 ?? 0), 0) / n;
  const avgTime = allResults.reduce((sum, r) => sum + r.timeMs, 0) / n;

  console.log(`OVERALL (n=${n})`);
  console.log(`  Dense@100: ${(denseRecallAt100 * 100).toFixed(1)}%`);
  if (EVAL_DENSE_ONLY) {
    console.log(`  Dense Hit@5: ${(denseHitAt5 * 100).toFixed(1)}%  Dense MRR@5: ${denseMrrAt5.toFixed(3)}`);
  }
  if (HYBRID_ALPHA !== null && !Number.isNaN(HYBRID_ALPHA)) {
    console.log(`  Hybrid(a=${HYBRID_ALPHA.toFixed(2)}) Hit@5: ${(hybridHitAt5 * 100).toFixed(1)}%  Hybrid MRR@5: ${hybridMrrAt5.toFixed(3)}`);
  }
  console.log(`  Hit@1:     ${(hitAt1 * 100).toFixed(1)}%`);
  console.log(`  Hit@5:     ${(hitAt5 * 100).toFixed(1)}%`);
  console.log(`  MRR@5:     ${mrrAt5.toFixed(3)}`);
  console.log(`  Avg query: ${avgTime.toFixed(1)}ms`);
  console.log();

  // By difficulty
  const byDiff = new Map<string, QueryResult[]>();
  for (const r of allResults) {
    const d = r.difficulty || "unknown";
    if (!byDiff.has(d)) byDiff.set(d, []);
    byDiff.get(d)!.push(r);
  }

  console.log("BY DIFFICULTY");
  for (const [diff, results] of byDiff) {
    const dn = results.length;
    const h1 = results.filter(r => r.hitAt1).length / dn;
    const h5 = results.filter(r => r.hitAt5).length / dn;
    const dr = results.filter(r => r.denseRecallAt100).length / dn;
    const m = results.reduce((sum, r) => sum + r.mrrAt5, 0) / dn;
    const dh5 = results.filter(r => r.denseHitAt5).length / dn;
    const dm = results.reduce((sum, r) => sum + r.denseMrrAt5, 0) / dn;
    const densePart = EVAL_DENSE_ONLY ? `  Dense@5: ${(dh5 * 100).toFixed(1).padStart(5)}%  dMRR@5: ${dm.toFixed(3)}` : "";
    const hh5 = results.filter(r => r.hybridHitAt5).length / dn;
    const hm = results.reduce((sum, r) => sum + (r.hybridMrrAt5 ?? 0), 0) / dn;
    const hybridPart = HYBRID_ALPHA !== null ? `  Hybrid@5: ${(hh5 * 100).toFixed(1).padStart(5)}%  hMRR@5: ${hm.toFixed(3)}` : "";
    console.log(`  ${diff.padEnd(10)} Dense@100: ${(dr * 100).toFixed(1).padStart(5)}%${densePart}${hybridPart}  Hit@1: ${(h1 * 100).toFixed(1).padStart(5)}%  Hit@5: ${(h5 * 100).toFixed(1).padStart(5)}%  MRR@5: ${m.toFixed(3)}  (n=${dn})`);
  }
  console.log();

  // By repo
  console.log("BY REPO");
  for (const [repo, repoRows] of byRepo) {
    const results = allResults.filter(r => r.repo === repo);
    if (results.length === 0) continue;
    const rn = results.length;
    const h1 = results.filter(r => r.hitAt1).length / rn;
    const h5 = results.filter(r => r.hitAt5).length / rn;
    const dr = results.filter(r => r.denseRecallAt100).length / rn;
    const m = results.reduce((sum, r) => sum + r.mrrAt5, 0) / rn;
    const dh5 = results.filter(r => r.denseHitAt5).length / rn;
    const dm = results.reduce((sum, r) => sum + r.denseMrrAt5, 0) / rn;
    const densePart = EVAL_DENSE_ONLY ? `  Dense@5: ${(dh5 * 100).toFixed(1).padStart(5)}%  dMRR@5: ${dm.toFixed(3)}` : "";
    const hh5 = results.filter(r => r.hybridHitAt5).length / rn;
    const hm = results.reduce((sum, r) => sum + (r.hybridMrrAt5 ?? 0), 0) / rn;
    const hybridPart = HYBRID_ALPHA !== null ? `  Hybrid@5: ${(hh5 * 100).toFixed(1).padStart(5)}%  hMRR@5: ${hm.toFixed(3)}` : "";
    console.log(`  ${repo.padEnd(12)} Dense@100: ${(dr * 100).toFixed(1).padStart(5)}%${densePart}${hybridPart}  Hit@1: ${(h1 * 100).toFixed(1).padStart(5)}%  Hit@5: ${(h5 * 100).toFixed(1).padStart(5)}%  MRR@5: ${m.toFixed(3)}  (n=${rn})`);
  }
  console.log();

  // Timing summary
  console.log("TIMING");
  console.log(`  Total index time: ${totalIndexTime.toFixed(0)}ms`);
  console.log(`  Total query time: ${totalQueryTime.toFixed(0)}ms`);
  console.log(`  Throughput: ${(1000 / avgTime).toFixed(1)} queries/sec`);
}

main().catch(console.error);
