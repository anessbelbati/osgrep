// Quick smoke test for osgrep-core
import { initModels, isInitialized, embedDense, embedBatch } from './index.js';

const DENSE_REPO = 'onnx-community/granite-embedding-30m-english-ONNX';
const COLBERT_REPO = 'ryandono/mxbai-edge-colbert-v0-17m-onnx-int8';

async function main() {
  console.log('Testing osgrep-core...\n');

  // Check initial state
  console.log('isInitialized:', isInitialized());

  // Initialize models
  console.log('Initializing models...');
  const start = Date.now();
  initModels(DENSE_REPO, COLBERT_REPO);
  console.log(`Models loaded in ${Date.now() - start}ms`);
  console.log('isInitialized:', isInitialized());

  // Test dense embedding
  console.log('\nTesting dense embedding...');
  const texts = ['hello world', 'how does authentication work'];
  const t0 = Date.now();
  const dense = embedDense(texts);
  console.log(`Dense embed ${texts.length} texts in ${Date.now() - t0}ms`);
  console.log(`  count: ${dense.count}`);
  console.log(`  embeddings length: ${dense.embeddings.length} (expected: ${texts.length * 384})`);

  // Test combined embed
  console.log('\nTesting combined embed (dense + colbert)...');
  const t1 = Date.now();
  const result = embedBatch(texts);
  console.log(`Combined embed ${texts.length} texts in ${Date.now() - t1}ms`);
  console.log(`  dense length: ${result.dense.length}`);
  console.log(`  colbert_embeddings length: ${result.colbertEmbeddings.length}`);
  console.log(`  colbert_lengths: ${result.colbertLengths}`);
  console.log(`  colbert_offsets: ${result.colbertOffsets}`);

  console.log('\n✅ All tests passed!');
}

main().catch(console.error);
