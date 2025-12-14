import { cloneRepoIfMissing, loadRepoChunks, walkFiles, isCodeFile } from "./_util.js";

// @ts-ignore - generated at build time
import { initColbertOrt, colbertRerankOrt } from "../index.js";

const REPO_URL = "https://github.com/sst/opencode.git";
const REPO_DIR = "./opencode";
const MODEL_REPO = "ryandono/osgrep-17m-v1-onnx";
const HIDDEN_SIZE = 48;  // osgrep-17m has 48-dim embeddings
const NUM_CANDIDATES = 100;  // Rerank top 100 candidates
const TOP_K = 5;  // Return top 5 results

async function main() {
  console.log("=== ColBERT Rerank Benchmark (ONNX Runtime) ===\n");

  // 1. Ensure repo exists
  cloneRepoIfMissing(REPO_URL, REPO_DIR);

  // 2. Load chunks
  console.log("Loading code files and chunking...");
  const files = walkFiles(REPO_DIR);
  const codeFiles = files.filter(isCodeFile);
  const allChunks = loadRepoChunks(REPO_DIR, 80);

  console.log(`  Files found: ${files.length}`);
  console.log(`  Code files: ${codeFiles.length}`);
  console.log(`  Total chunks: ${allChunks.length}`);
  console.log(`  Candidates per query: ${NUM_CANDIDATES}`);
  console.log(`  Top-K: ${TOP_K}`);
  console.log();

  // 3. Initialize model
  console.log(`Initializing ColBERT ORT model from: ${MODEL_REPO}`);
  const initStart = performance.now();
  initColbertOrt(MODEL_REPO, HIDDEN_SIZE);
  const initTime = performance.now() - initStart;
  console.log(`Model initialized in ${initTime.toFixed(0)}ms\n`);

  // 4. Test queries
  const queries = [
    "where is authentication handled",
    "how does the API route requests",
    "database connection and query execution",
    "error handling and logging",
    "configuration and environment variables",
  ];

  // Take first NUM_CANDIDATES chunks as simulated candidates
  const candidates = allChunks.slice(0, NUM_CANDIDATES);

  console.log("Starting benchmark...\n");

  let totalRerankTime = 0;
  const results: { query: string; topIndices: number[]; topScores: number[]; timeMs: number }[] = [];

  for (const query of queries) {
    const startTime = performance.now();
    const result = colbertRerankOrt(query, candidates, TOP_K);
    const elapsed = performance.now() - startTime;
    totalRerankTime += elapsed;

    results.push({
      query,
      topIndices: Array.from(result.indices),
      topScores: Array.from(result.scores),
      timeMs: elapsed,
    });

    console.log(`Query: "${query}"`);
    console.log(`  Time: ${elapsed.toFixed(1)}ms`);
    console.log(`  Top indices: [${result.indices.slice(0, 3).join(", ")}...]`);
    console.log(`  Top scores: [${result.scores.slice(0, 3).map((s: number) => s.toFixed(3)).join(", ")}...]`);
    console.log();
  }

  const avgTime = totalRerankTime / queries.length;

  console.log("=== Summary ===");
  console.log(`  Queries: ${queries.length}`);
  console.log(`  Candidates per query: ${NUM_CANDIDATES}`);
  console.log(`  Total rerank time: ${totalRerankTime.toFixed(1)}ms`);
  console.log(`  Avg time per query: ${avgTime.toFixed(1)}ms`);
  console.log(`  Throughput: ${(1000 / avgTime).toFixed(1)} queries/sec`);
}

main().catch(console.error);
