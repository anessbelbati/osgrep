import { inner } from "simsimd";

/**
 * Computes the MaxSim score between a Query and a Document.
 * * Late Interaction mechanism:
 * 1. For every token in the Query...
 * 2. Find the maximum similarity (dot product) with ANY token in the Document.
 * 3. Sum those maximums.
 * * @param queryEmbeddings - Array of vectors [seq_len_q, dim]
 * @param docEmbeddings - Array of vectors [seq_len_d, dim]
 */
export function maxSim(
  queryEmbeddings: number[][] | Float32Array[],
  docEmbeddings: number[][] | Float32Array[],
): number {
  if (queryEmbeddings.length === 0 || docEmbeddings.length === 0) {
    return 0;
  }

  // Ensure inputs are Float32Arrays for simsimd
  const qVecs = queryEmbeddings.map((v) =>
    v instanceof Float32Array ? v : new Float32Array(v),
  );
  const dVecs = docEmbeddings.map((v) =>
    v instanceof Float32Array ? v : new Float32Array(v),
  );

  // We assume normalized embeddings (dot product = cosine similarity)
  let totalScore = 0;

  // Iterate over every query token
  for (const qVec of qVecs) {
    let maxDotProduct = -Infinity;

    // Compare against every document token
    for (const dVec of dVecs) {
      let dot: number;

      // Safety: Use the smaller dimension to avoid NaN if models mismatch
      if (qVec.length !== dVec.length) {
        const limit = Math.min(qVec.length, dVec.length);
        dot = inner(qVec.subarray(0, limit), dVec.subarray(0, limit));
      } else {
        dot = inner(qVec, dVec);
      }

      if (dot > maxDotProduct) {
        maxDotProduct = dot;
      }
    }

    // Sum the best matches
    totalScore += maxDotProduct;
  }

  return totalScore;
}

/**
 * Computes the Cosine Similarity between two vectors.
 */
export function cosineSim(
  a: number[] | Float32Array,
  b: number[] | Float32Array,
): number {
  const aVec = a instanceof Float32Array ? a : new Float32Array(a);
  const bVec = b instanceof Float32Array ? b : new Float32Array(b);

  const dim = Math.min(aVec.length, bVec.length);
  if (aVec.length !== bVec.length) {
    return inner(aVec.subarray(0, dim), bVec.subarray(0, dim));
  }
  return inner(aVec, bVec);
}