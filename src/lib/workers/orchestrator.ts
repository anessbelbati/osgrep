/**
 * Orchestrator: Coordinates file processing and embedding
 *
 * This module handles:
 * - File reading and validation
 * - Chunking via tree-sitter
 * - Embedding via native Rust code
 *
 * No worker pool needed - Rust ONNX Runtime is fast and stable.
 */

import * as path from "node:path";
import { v4 as uuidv4 } from "uuid";
import { CONFIG } from "../../config";
import {
  buildAnchorChunk,
  type ChunkWithContext,
  formatChunkText,
  TreeSitterChunker,
} from "../index/chunker";
import { embedBatch, initNative, encodeQueryColbert } from "../native";
import type { PreparedChunk, VectorRecord } from "../store/types";
import {
  computeBufferHash,
  hasNullByte,
  isIndexableFile,
  readFileSnapshot,
} from "../utils/file-utils";

// =============================================================================
// Types
// =============================================================================

export type ProcessFileInput = {
  path: string;
  absolutePath?: string;
};

export type ProcessFileResult = {
  vectors: VectorRecord[];
  hash: string;
  mtimeMs: number;
  size: number;
  shouldDelete?: boolean;
};

export type RerankDoc = {
  colbert: Buffer | Int8Array | number[];
  scale: number;
  token_ids?: number[];
};

// =============================================================================
// Orchestrator
// =============================================================================

const PROJECT_ROOT = process.env.OSGREP_PROJECT_ROOT
  ? path.resolve(process.env.OSGREP_PROJECT_ROOT)
  : process.cwd();

export class WorkerOrchestrator {
  private chunker = new TreeSitterChunker();
  private initPromise: Promise<void> | null = null;

  private async ensureReady() {
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      await Promise.all([this.chunker.init(), initNative()]);
    })().finally(() => {
      this.initPromise = null;
    });

    return this.initPromise;
  }

  private async chunkFile(
    pathname: string,
    content: string
  ): Promise<ChunkWithContext[]> {
    await this.ensureReady();
    const { chunks: parsedChunks, metadata } = await this.chunker.chunk(
      pathname,
      content
    );

    const anchorChunk = buildAnchorChunk(pathname, content, metadata);
    const baseChunks = anchorChunk
      ? [anchorChunk, ...parsedChunks]
      : parsedChunks;

    return baseChunks.map((chunk, idx) => {
      const chunkWithContext = chunk as ChunkWithContext;
      return {
        ...chunkWithContext,
        context: Array.isArray(chunkWithContext.context)
          ? chunkWithContext.context
          : [],
        chunkIndex:
          typeof chunkWithContext.chunkIndex === "number"
            ? chunkWithContext.chunkIndex
            : anchorChunk
              ? idx - 1
              : idx,
        isAnchor:
          chunkWithContext.isAnchor === true ||
          (anchorChunk ? idx === 0 : false),
        imports: metadata.imports,
      };
    });
  }

  private toPreparedChunks(
    filePath: string,
    hash: string,
    chunks: ChunkWithContext[]
  ): PreparedChunk[] {
    const texts = chunks.map((chunk) => formatChunkText(chunk, filePath));
    const prepared: PreparedChunk[] = [];

    for (let i = 0; i < texts.length; i++) {
      const chunk = chunks[i];
      const { content, displayText } = texts[i];
      const prev = texts[i - 1]?.displayText;
      const next = texts[i + 1]?.displayText;

      prepared.push({
        id: uuidv4(),
        path: filePath,
        hash,
        content,
        display_text: displayText,
        context_prev: typeof prev === "string" ? prev : undefined,
        context_next: typeof next === "string" ? next : undefined,
        start_line: chunk.startLine,
        end_line: chunk.endLine,
        chunk_index: chunk.chunkIndex,
        is_anchor: chunk.isAnchor === true,
        chunk_type: typeof chunk.type === "string" ? chunk.type : undefined,
        complexity: chunk.complexity,
        is_exported: chunk.isExported,
        defined_symbols: chunk.definedSymbols,
        referenced_symbols: chunk.referencedSymbols,
        role: chunk.role,
        parent_symbol: chunk.parentSymbol,
      });
    }

    return prepared;
  }

  async processFile(
    input: ProcessFileInput,
    onProgress?: () => void
  ): Promise<ProcessFileResult> {
    const absolutePath = path.isAbsolute(input.path)
      ? input.path
      : input.absolutePath
        ? input.absolutePath
        : path.join(PROJECT_ROOT, input.path);

    const { buffer, mtimeMs, size } = await readFileSnapshot(absolutePath);
    const hash = computeBufferHash(buffer);

    if (!isIndexableFile(absolutePath, size)) {
      return { vectors: [], hash, mtimeMs, size, shouldDelete: true };
    }

    if (buffer.length === 0 || hasNullByte(buffer)) {
      return { vectors: [], hash, mtimeMs, size, shouldDelete: true };
    }

    onProgress?.();
    await this.ensureReady();
    onProgress?.();

    const content = buffer.toString("utf-8");
    const chunks = await this.chunkFile(input.path, content);
    onProgress?.();

    if (!chunks.length) return { vectors: [], hash, mtimeMs, size };

    const preparedChunks = this.toPreparedChunks(input.path, hash, chunks);

    // Embed all chunks via native Rust
    const hybrids = await embedBatch(
      preparedChunks.map((chunk) => chunk.content)
    );
    onProgress?.();

    const vectors: VectorRecord[] = preparedChunks.map((chunk, idx) => {
      const hybrid = hybrids[idx];
      return {
        ...chunk,
        vector: hybrid.dense,
        colbert: Buffer.from(hybrid.colbert),
        colbert_scale: 1, // Native returns pre-scaled INT8
        pooled_colbert_48d: undefined, // Can compute if needed
        doc_token_ids: undefined,
      };
    });

    onProgress?.();
    return { vectors, hash, mtimeMs, size };
  }

  async encodeQuery(text: string): Promise<{
    dense: number[];
    colbert: number[][];
    colbertDim: number;
    pooled_colbert_48d?: number[];
  }> {
    await this.ensureReady();

    // Get dense embedding
    const hybrids = await embedBatch([text]);
    const denseVector = hybrids[0].dense;

    // Get ColBERT query embedding
    const colbertFlat = await encodeQueryColbert(text);
    const dim = CONFIG.COLBERT_DIM;
    const seqLen = colbertFlat.length / dim;

    // Reshape to matrix
    const matrix: number[][] = [];
    for (let s = 0; s < seqLen; s++) {
      const row: number[] = [];
      for (let d = 0; d < dim; d++) {
        row.push(colbertFlat[s * dim + d]);
      }
      matrix.push(row);
    }

    // Compute pooled embedding (mean of tokens)
    const pooled = new Float32Array(dim);
    for (const row of matrix) {
      for (let d = 0; d < dim; d++) {
        pooled[d] += row[d];
      }
    }
    // Normalize
    let sumSq = 0;
    for (let d = 0; d < dim; d++) {
      pooled[d] /= matrix.length || 1;
      sumSq += pooled[d] * pooled[d];
    }
    const norm = Math.sqrt(sumSq);
    if (norm > 1e-9) {
      for (let d = 0; d < dim; d++) {
        pooled[d] /= norm;
      }
    }

    return {
      dense: Array.from(denseVector),
      colbert: matrix,
      colbertDim: dim,
      pooled_colbert_48d: Array.from(pooled),
    };
  }

  async rerank(input: {
    query: number[][];
    docs: RerankDoc[];
    colbertDim: number;
  }): Promise<number[]> {
    await this.ensureReady();

    // MaxSim scoring in TypeScript (simple, matches Rust behavior)
    const queryMatrix = input.query.map((row) =>
      row instanceof Float32Array ? row : new Float32Array(row)
    );

    return input.docs.map((doc) => {
      const col = doc.colbert;
      let colbert: Int8Array;

      if (col instanceof Int8Array) {
        colbert = col;
      } else if (Buffer.isBuffer(col)) {
        colbert = new Int8Array(col.buffer, col.byteOffset, col.byteLength);
      } else if (Array.isArray(col)) {
        colbert = new Int8Array(col);
      } else {
        colbert = new Int8Array(0);
      }

      const seqLen = Math.floor(colbert.length / input.colbertDim);

      // MaxSim: for each query token, find max similarity with doc tokens, sum
      let totalScore = 0;
      for (let q = 0; q < queryMatrix.length; q++) {
        const qRow = queryMatrix[q];
        let maxDot = -Infinity;

        for (let d = 0; d < seqLen; d++) {
          let dot = 0;
          for (let k = 0; k < input.colbertDim; k++) {
            // Dequantize INT8 back to float
            const docVal = (colbert[d * input.colbertDim + k] * doc.scale) / 127;
            dot += qRow[k] * docVal;
          }
          if (dot > maxDot) maxDot = dot;
        }

        if (maxDot > -Infinity) {
          totalScore += maxDot;
        }
      }

      return totalScore;
    });
  }
}

// =============================================================================
// Singleton for direct use (no worker pool needed)
// =============================================================================

let orchestrator: WorkerOrchestrator | null = null;

export function getOrchestrator(): WorkerOrchestrator {
  if (!orchestrator) {
    orchestrator = new WorkerOrchestrator();
  }
  return orchestrator;
}

export async function processFile(
  input: ProcessFileInput
): Promise<ProcessFileResult> {
  return getOrchestrator().processFile(input);
}

export async function encodeQuery(text: string) {
  return getOrchestrator().encodeQuery(text);
}

export async function rerank(input: {
  query: number[][];
  docs: RerankDoc[];
  colbertDim: number;
}) {
  return getOrchestrator().rerank(input);
}
