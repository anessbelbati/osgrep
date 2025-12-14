import { cloneRepoIfMissing, loadRepoChunks, walkFiles, isCodeFile } from "./_util.js";

// @ts-ignore - generated at build time
import { initColbertOrt, colbertPreindexDocs, colbertRerankPreindexed } from "../index.js";

const REPO_URL = "https://github.com/sst/opencode.git";
const REPO_DIR = "./opencode";
const MODEL_REPO = "ryandono/osgrep-17m-v1-onnx";
const HIDDEN_SIZE = 48;
const NUM_CANDIDATES = 100;
const TOP_K = 5;

async function main() {
  console.log("=== ColBERT Pre-indexed Rerank Benchmark ===\n");

  // 1. Load repo
  cloneRepoIfMissing(REPO_URL, REPO_DIR);

  console.log("Loading code files and chunking...");
  const files = walkFiles(REPO_DIR);
  const codeFiles = files.filter(isCodeFile);
  const allChunks = loadRepoChunks(REPO_DIR, 80);

  console.log(`  Files found: ${files.length}`);
  console.log(`  Code files: ${codeFiles.length}`);
  console.log(`  Total chunks: ${allChunks.length}`);
  console.log();

  // 2. Initialize model
  console.log(`Initializing ColBERT ORT model from: ${MODEL_REPO}`);
  initColbertOrt(MODEL_REPO, HIDDEN_SIZE);
  console.log("Model initialized.\n");

  // 3. Pre-index first 500 chunks (simulating index time)
  const indexChunks = allChunks.slice(0, 500);
  console.log(`Pre-indexing ${indexChunks.length} chunks (INDEX TIME)...`);
  const indexStart = performance.now();
  const numIndexed = colbertPreindexDocs(indexChunks);
  const indexTime = performance.now() - indexStart;
  console.log(`  Indexed ${numIndexed} chunks in ${indexTime.toFixed(0)}ms`);
  console.log(`  Rate: ${((indexChunks.length / indexTime) * 1000).toFixed(1)} chunks/sec`);
  console.log();

  // 4. Simulate dense retrieval returning top 100 indices
  // In real use, these would come from your dense ORT search
  const candidateIndices = Array.from({ length: NUM_CANDIDATES }, (_, i) => i);

  // 5. Test queries (QUERY TIME)
  const queries = [
    "where is authentication handled",
    "how does the API route requests",
    "database connection and query execution",
    "error handling and logging",
    "configuration and environment variables",
  ];

  console.log(`Starting QUERY TIME benchmark (${NUM_CANDIDATES} candidates per query)...\n`);

  let totalRerankTime = 0;
  const results: { query: string; timeMs: number }[] = [];

  for (const query of queries) {
    const startTime = performance.now();
    const result = colbertRerankPreindexed(query, candidateIndices, TOP_K);
    const elapsed = performance.now() - startTime;
    totalRerankTime += elapsed;

    results.push({ query, timeMs: elapsed });

    console.log(`Query: "${query}"`);
    console.log(`  Time: ${elapsed.toFixed(1)}ms`);
    console.log(`  Top indices: [${result.indices.slice(0, 3).join(", ")}...]`);
    console.log(`  Top scores: [${result.scores.slice(0, 3).map((s: number) => s.toFixed(3)).join(", ")}...]`);
    console.log();
  }

  const avgTime = totalRerankTime / queries.length;

  console.log("=== QUERY TIME Summary ===");
  console.log(`  Queries: ${queries.length}`);
  console.log(`  Candidates per query: ${NUM_CANDIDATES}`);
  console.log(`  Total rerank time: ${totalRerankTime.toFixed(1)}ms`);
  console.log(`  Avg time per query: ${avgTime.toFixed(1)}ms`);
  console.log(`  Throughput: ${(1000 / avgTime).toFixed(1)} queries/sec`);
  console.log();

  if (avgTime < 25) {
    console.log(`  ✓ TARGET MET: <25ms per query!`);
  } else {
    console.log(`  ✗ Target not met (want <25ms, got ${avgTime.toFixed(1)}ms)`);
  }
}

main().catch(console.error);
