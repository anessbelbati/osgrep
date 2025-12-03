import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { parentPort } from "node:worker_threads";
import * as ort from "onnxruntime-node";
import { env } from "@huggingface/transformers";
import { CONFIG, PATHS } from "../../config";
import { MODEL_IDS } from "../../config";
import { ColBERTTokenizer } from "./colbert-tokenizer";

function resolveThreadCount(): number {
  const fromEnv = Number.parseInt(process.env.OSGREP_THREADS ?? "", 10);
  if (Number.isFinite(fromEnv) && fromEnv > 0) return fromEnv;
  if (process.env.OSGREP_LOW_IMPACT === "1") return 1;
  const cores = os.cpus().length || 1;
  return Math.max(1, Math.min(cores - 1, 4));
}

const CACHE_DIR = PATHS.models;
const NUM_THREADS = resolveThreadCount();
const LOG_MODELS =
  process.env.OSGREP_DEBUG_MODELS === "1" ||
  process.env.OSGREP_DEBUG_MODELS === "true";
const log = (...args: unknown[]) => {
  if (LOG_MODELS) console.log(...args);
};

// Configure ONNX Runtime threading
// Note: onnxruntime-node configuration is done via session options, not global env like transformers.js

// Configure transformers.js env for tokenizer downloads
env.cacheDir = CACHE_DIR;
env.allowLocalModels = true;
env.allowRemoteModels = true; // Allow downloading tokenizer

// Try to find local models directory
const PROJECT_ROOT = process.cwd();
const LOCAL_MODELS = path.join(PROJECT_ROOT, "models");
if (fs.existsSync(LOCAL_MODELS)) {
  env.localModelPath = LOCAL_MODELS;
  log(`Worker: Using local models from ${LOCAL_MODELS}`);
}

type EmbedOutput = {
  data: Float32Array | number[];
  dims?: number[];
};

class EmbeddingWorker {
  // Dense pipeline (transformers.js)
  private embedPipe: any = null;

  // ColBERT components (Native ONNX)
  private colbertSession: ort.InferenceSession | null = null;
  private colbertTokenizer: ColBERTTokenizer | null = null;

  private embedModelId = MODEL_IDS.embed;
  private colbertModelId = MODEL_IDS.colbert;
  private readonly vectorDimensions = CONFIG.VECTOR_DIMENSIONS;
  private initPromise: Promise<void> | null = null;

  async initialize() {
    if (this.embedPipe && this.colbertSession) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      try {
        log(`Worker: Loading models...`);

        // 1. Load Dense Model (transformers.js)
        if (!this.embedPipe) {
          // Force CPU for stability and consistency
          // @ts-ignore
          const useCpu = env.backends?.onnx?.wasm?.numThreads !== 1;

          if (!this.embedPipe) {
            const { pipeline } = await import("@huggingface/transformers");
            this.embedPipe = await pipeline(
              "feature-extraction",
              this.embedModelId,
              {
                dtype: "q4",
                device: "cpu",
              }
            );
          }
        }

        // 2. Load ColBERT Model (Native ONNX)
        if (!this.colbertSession) {
          // Initialize Tokenizer
          this.colbertTokenizer = new ColBERTTokenizer();

          let modelPath = "";
          let tokenizerPath = "";

          // Check experiment folder first (DEV MODE)
          const devModelPath = "/Users/ryandonofrio/Desktop/osgrep2/Archive-1/models/distilled_colbert/onnx/model.onnx";
          const devTokenizerPath = "/Users/ryandonofrio/Desktop/osgrep2/Archive-1/models/distilled_colbert";

          if (fs.existsSync(devModelPath)) {
            modelPath = devModelPath;
            tokenizerPath = devTokenizerPath;
            log(`Worker: Using DEV model from ${modelPath}`);
          } else {
            // Fallback to ~/.osgrep/models
            const prodPath = path.join(CACHE_DIR, this.colbertModelId, "onnx", "model.onnx");
            if (fs.existsSync(prodPath)) {
              modelPath = prodPath;
              tokenizerPath = path.join(CACHE_DIR, this.colbertModelId);
            }
          }

          if (!modelPath) {
            throw new Error(`ColBERT ONNX model not found. Expected at ${devModelPath} or ~/.osgrep/models/${this.colbertModelId}/onnx/model.onnx`);
          }

          await this.colbertTokenizer.init(tokenizerPath);

          log(`Worker: Loading native ONNX session from ${modelPath}`);

          const sessionOptions: ort.InferenceSession.SessionOptions = {
            executionProviders: ["cpu"],
            intraOpNumThreads: NUM_THREADS,
            interOpNumThreads: 1,
            graphOptimizationLevel: "all",
          };

          this.colbertSession = await ort.InferenceSession.create(modelPath, sessionOptions);
        }

        log("Worker: Models loaded.");
      } finally {
        this.initPromise = null;
      }
    })();

    return this.initPromise;
  }

  private toDenseVectors(output: EmbedOutput): number[][] {
    const dims = output.dims || [1, this.vectorDimensions];
    const data = output.data;

    if (dims.length === 3) {
      const [batch, seqLen, dim] = dims;
      const embeddings: number[][] = [];
      for (let i = 0; i < batch; i++) {
        const start = i * seqLen * dim;
        const cls = data.slice(start, start + dim);
        const vec = Array.from(cls as ArrayLike<number>).slice(0, this.vectorDimensions);
        while (vec.length < this.vectorDimensions) vec.push(0);
        embeddings.push(vec);
      }
      return embeddings;
    }

    let batchSize = 1;
    let hiddenSize = this.vectorDimensions;

    if (dims.length === 2) {
      batchSize = dims[0];
      hiddenSize = dims[1];
    } else if (dims.length === 1) {
      hiddenSize = dims[0];
    }

    const embeddings: number[][] = [];
    for (let i = 0; i < batchSize; i++) {
      const start = i * hiddenSize;
      const slice = output.data.slice(
        start,
        start + Math.min(this.vectorDimensions, hiddenSize),
      );
      const vec: number[] = Array.from(slice as ArrayLike<number>);
      if (vec.length < this.vectorDimensions) {
        vec.push(...Array(this.vectorDimensions - vec.length).fill(0));
      }
      embeddings.push(vec);
    }

    return embeddings;
  }

  async computeHybrid(
    texts: string[],
  ): Promise<Array<{ dense: number[]; colbert: Buffer; scale: number }>> {
    if (!this.embedPipe || !this.colbertSession || !this.colbertTokenizer) await this.initialize();

    // 1. Dense Embedding (transformers.js)
    const denseOut = await this.embedPipe(texts, {
      pooling: "cls",
      normalize: true,
      truncation: true,
      max_length: 256,
    });
    const denseVectors = this.toDenseVectors(denseOut);

    // 2. ColBERT Embedding (Native ONNX)
    const results: Array<{ dense: number[]; colbert: Buffer; scale: number }> = [];

    // Process in batches
    const BATCH_SIZE = 32;
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batchTexts = texts.slice(i, i + BATCH_SIZE);
      const batchDense = denseVectors.slice(i, i + BATCH_SIZE);

      // 1. Tokenize all in batch
      const encodedBatch = await Promise.all(
        batchTexts.map((t) => this.colbertTokenizer!.encodeDoc(t)),
      );

      // 2. Pad to max length in this batch
      const maxLen = Math.max(...encodedBatch.map((e) => e.input_ids.length));
      const batchInputIds: bigint[] = [];
      const batchAttentionMask: bigint[] = [];

      for (const encoded of encodedBatch) {
        const len = encoded.input_ids.length;
        const padding = maxLen - len;

        // Add input_ids (pad with 0/pad_token if needed, but usually pad token is specific)
        // ColBERTTokenizer uses specialTokenIds.pad which is 50283
        // We need to access it or just assume 0 if not available, but let's use the tokenizer's pad id if we can.
        // Actually, ColBERTTokenizer doesn't expose it easily, but we can see it's 50283 in the file.
        // Let's check if we can access it. We can't easily from here without changing tokenizer.
        // However, the tokenizer implementation has `specialTokenIds` private.
        // Let's assume 50283 for now or just 0 (which is often ignored by attention mask anyway).
        // Wait, `ColBERTTokenizer` has `specialTokenIds` private.
        // Let's use 0 for padding and ensure attention mask is 0.
        const padId = BigInt(50283); // Standard XLM-RoBERTa pad token

        batchInputIds.push(...encoded.input_ids);
        if (padding > 0) {
          batchInputIds.push(...Array(padding).fill(padId));
        }

        batchAttentionMask.push(...encoded.attention_mask);
        if (padding > 0) {
          batchAttentionMask.push(...Array(padding).fill(BigInt(0)));
        }
      }

      // 3. Run Inference
      const feeds = {
        input_ids: new ort.Tensor("int64", BigInt64Array.from(batchInputIds), [
          batchTexts.length,
          maxLen,
        ]),
        attention_mask: new ort.Tensor(
          "int64",
          BigInt64Array.from(batchAttentionMask),
          [batchTexts.length, maxLen],
        ),
      };

      const sessionOut = await this.colbertSession!.run(feeds);
      const output = sessionOut[this.colbertSession!.outputNames[0]];

      // 4. Process Batch Output
      const data = output.data as Float32Array;
      const [batch, seq, dim] = output.dims; // [batch, seq, 48]

      for (let b = 0; b < batch; b++) {
        const batchOffset = b * seq * dim;
        // We only care about the actual tokens, not the padding.
        // The original length of this doc:
        const originalLen = encodedBatch[b].input_ids.length;

        // Extract, Normalize, Quantize
        // We can reuse the same buffer logic but we need to be careful about the slice.
        // The output includes embeddings for padding tokens too, we should ignore them?
        // ColBERT usually ignores them or they are masked out.
        // Let's process up to originalLen.

        const docEmbeddings: number[] = [];
        let maxVal = 0;

        for (let s = 0; s < originalLen; s++) {
          const offset = batchOffset + s * dim;
          let sumSq = 0;
          for (let d = 0; d < dim; d++) {
            const val = data[offset + d];
            sumSq += val * val;
          }
          const norm = Math.sqrt(sumSq);

          for (let d = 0; d < dim; d++) {
            let val = data[offset + d];
            if (norm > 1e-9) val /= norm;
            docEmbeddings.push(val);
            if (Math.abs(val) > maxVal) maxVal = Math.abs(val);
          }
        }

        if (maxVal === 0) maxVal = 1;

        const int8Array = new Int8Array(docEmbeddings.length);
        for (let k = 0; k < docEmbeddings.length; k++) {
          int8Array[k] = Math.max(
            -127,
            Math.min(127, Math.round((docEmbeddings[k] / maxVal) * 127)),
          );
        }

        results.push({
          dense: batchDense[b] ?? [],
          colbert: Buffer.from(int8Array),
          scale: maxVal,
        });
      }
    }

    return results;
  }

  async encodeQuery(
    text: string,
  ): Promise<{ dense: number[]; colbert: number[][]; colbertDim: number }> {
    if (!this.embedPipe || !this.colbertSession || !this.colbertTokenizer) await this.initialize();

    // 1. Dense Embedding
    const denseOut = await this.embedPipe([text], {
      pooling: "cls",
      normalize: true,
      truncation: true,
      max_length: 256,
    });
    const denseVector = this.toDenseVectors(denseOut)[0] ?? [];

    // 2. ColBERT Embedding (Native ONNX)
    const encoded = await this.colbertTokenizer!.encodeQuery(text);

    const feeds = {
      input_ids: new ort.Tensor('int64', encoded.input_ids, [1, encoded.input_ids.length]),
      attention_mask: new ort.Tensor('int64', encoded.attention_mask, [1, encoded.attention_mask.length])
    };

    const sessionOut = await this.colbertSession!.run(feeds);
    const output = sessionOut[this.colbertSession!.outputNames[0]];

    // Normalize (L2)
    const data = output.data as Float32Array;
    const [, seq, dim] = output.dims;

    const matrix: number[][] = [];

    for (let s = 0; s < seq; s++) {
      let sumSq = 0;
      const offset = s * dim;
      for (let d = 0; d < dim; d++) {
        const val = data[offset + d];
        sumSq += val * val;
      }
      const norm = Math.sqrt(sumSq);

      const row: number[] = [];
      if (norm > 1e-9) {
        for (let d = 0; d < dim; d++) {
          row.push(data[offset + d] / norm);
        }
      } else {
        for (let d = 0; d < dim; d++) {
          row.push(data[offset + d]);
        }
      }
      matrix.push(row);
    }

    return { dense: denseVector, colbert: matrix, colbertDim: dim };
  }
}

const worker = new EmbeddingWorker();

if (parentPort) {
  parentPort.on(
    "message",
    async (message: {
      id: string;
      type?: string;
      hybrid?: { texts: string[] };
      query?: { text: string };
    }) => {
      try {
        if (message.type === "shutdown") {
          process.exit(0);
          return;
        }

        if (message.hybrid) {
          const result = await worker.computeHybrid(message.hybrid.texts);
          const memory = process.memoryUsage();
          parentPort?.postMessage({ id: message.id, result, memory });
          return;
        }

        if (message.query) {
          const query = await worker.encodeQuery(message.query.text);
          const memory = process.memoryUsage();
          parentPort?.postMessage({ id: message.id, query, memory });
          return;
        }

        throw new Error("Unknown message type");
      } catch (error) {
        console.error("Worker error:", error);
        parentPort?.postMessage({
          id: message.id,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },
  );
}
