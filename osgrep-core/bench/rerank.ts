import { cloneRepoIfMissing, loadRepoChunks, pickCandidates } from "./_util";

// Import the native addon (will be built by napi)
// @ts-ignore - generated at build time
import { initColbert, colbertRerankChecksum } from "../index";

const REPO_URL = "https://github.com/sst/opencode.git";
const REPO_DIR = "./opencode";
const MODEL_REPO = "mixedbread-ai/mxbai-edge-colbert-v0-17m";
const NUM_CANDIDATES = 100;
const TOP_K = 5;
const QUERY = "where is auth handled";
const LINES_PER_CHUNK = 80;

async function main() {
  console.log("=== ColBERT Rerank Benchmark ===\n");

  // 1. Ensure repo exists
  cloneRepoIfMissing(REPO_URL, REPO_DIR);

  // 2. Load chunks and pick candidates
  console.log("Loading code files and selecting candidates...");
  const allChunks = loadRepoChunks(REPO_DIR, LINES_PER_CHUNK);
  const candidates = pickCandidates(allChunks, NUM_CANDIDATES);

  console.log(`  Total chunks available: ${allChunks.length}`);
  console.log(`  Candidates selected: ${candidates.length}`);
  console.log(`  Query: "${QUERY}"`);
  console.log(`  Top-K: ${TOP_K}`);
  console.log();

  // 3. Initialize model
  console.log(`Initializing ColBERT model: ${MODEL_REPO}`);
  initColbert(MODEL_REPO);
  console.log("Model initialized.\n");

  // 4. Benchmark reranking
  console.log("Running rerank benchmark...");
  const startTime = performance.now();

  const result = colbertRerankChecksum(QUERY, candidates, TOP_K);

  const endTime = performance.now();
  const totalMs = endTime - startTime;

  console.log("\n=== Results ===");
  console.log(`  Total time: ${totalMs.toFixed(2)} ms`);
  console.log(`  Checksum: ${result.checksum.toFixed(6)} (for validation)`);
  console.log();
  console.log("  Top-5 results:");
  for (let i = 0; i < result.indices.length; i++) {
    const idx = result.indices[i];
    const score = result.scores[i];
    const preview = candidates[idx].slice(0, 80).replace(/\n/g, " ").trim();
    console.log(`    [${i + 1}] idx=${idx}, score=${score.toFixed(4)}`);
    console.log(`        "${preview}..."`);
  }
}

main().catch(console.error);
