import { cloneRepoIfMissing, loadRepoChunks, walkFiles, isCodeFile } from "./_util.js";

// Import the native addon (will be built by napi)
// @ts-ignore - generated at build time
import { initDenseOrt, denseEmbedChecksumOrt } from "../index.js";

const REPO_URL = "https://github.com/sst/opencode.git";
const REPO_DIR = "./opencode";
const MODEL_REPO = "onnx-community/granite-embedding-30m-english-ONNX";
const HIDDEN_SIZE = 384;  // granite-embedding-30m-english has 384 dim
const BATCH_SIZE = 64;  // Larger batch for throughput
const LINES_PER_CHUNK = 80;  // Original chunk size

async function main() {
  console.log("=== Dense Embedding Benchmark (ONNX Runtime) ===\n");

  // 1. Ensure repo exists
  cloneRepoIfMissing(REPO_URL, REPO_DIR);

  // 2. Load chunks
  console.log("Loading code files and chunking...");
  const files = walkFiles(REPO_DIR);
  const codeFiles = files.filter(isCodeFile);
  const chunks = loadRepoChunks(REPO_DIR, LINES_PER_CHUNK);

  console.log(`  Files found: ${files.length}`);
  console.log(`  Code files: ${codeFiles.length}`);
  console.log(`  Total chunks: ${chunks.length}`);
  console.log(`  Batch size: ${BATCH_SIZE}`);
  console.log();

  // 3. Initialize model from HuggingFace ONNX repo
  console.log(`Initializing ORT model from: ${MODEL_REPO}`);
  initDenseOrt(MODEL_REPO, HIDDEN_SIZE);
  console.log("Model initialized.\n");

  // 4. Benchmark encoding
  console.log("Starting benchmark...");
  const startTime = performance.now();
  let totalChecksum = 0;
  let batchCount = 0;

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const checksum = denseEmbedChecksumOrt(batch, true);
    totalChecksum += checksum;
    batchCount++;

    // Progress indicator every batch
    const elapsed = (performance.now() - startTime) / 1000;
    const chunksProcessed = Math.min(i + BATCH_SIZE, chunks.length);
    const rate = chunksProcessed / elapsed;
    process.stdout.write(`\r  Processed ${chunksProcessed}/${chunks.length} chunks (${rate.toFixed(1)} chunks/sec)`);
  }

  const endTime = performance.now();
  const totalSeconds = (endTime - startTime) / 1000;
  const chunksPerSec = chunks.length / totalSeconds;

  console.log("\n");
  console.log("=== Results ===");
  console.log(`  File count: ${codeFiles.length}`);
  console.log(`  Chunk count: ${chunks.length}`);
  console.log(`  Total time: ${totalSeconds.toFixed(2)} seconds`);
  console.log(`  Throughput: ${chunksPerSec.toFixed(1)} chunks/sec`);
  console.log(`  Checksum: ${totalChecksum.toFixed(6)} (for validation)`);
}

main().catch(console.error);
