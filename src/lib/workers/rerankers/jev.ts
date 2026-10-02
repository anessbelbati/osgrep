/**
 * Jev Reranker Provider (TypeSafe's Jev, over its System One protocol)
 *
 * One request per search: every candidate goes in as a passage with a
 * four-level relevance question of its own, and a passage's score is the
 * expected level, scaled to 0..1.
 * Configure via environment variables:
 *   - JEV_API_KEY: Key for the endpoint (optional on OpenRouter, where
 *     OPENROUTER_API_KEY and then QWEN_API_KEY are used)
 *   - JEV_API_ENDPOINT: Custom endpoint (optional, defaults to OpenRouter)
 *   - JEV_MODEL: Model name (optional)
 */

import { CLOUD_API } from "../../../config";
import type { RerankWithTextInput, RerankWithTextResult } from "./zerank";

const LOG_MODELS =
  process.env.OSGREP_DEBUG_MODELS === "1" ||
  process.env.OSGREP_DEBUG_MODELS === "true";
const log = (...args: unknown[]) => {
  if (LOG_MODELS) console.log(...args);
};

// The wording Jev was measured with as a reranker ("jev-score-batch" in
// github.com/anessbelbati/jev-rerank-bench). Scores are only comparable with
// that benchmark while it stays word for word.
const RUBRIC = [
  "The passage is off-topic for the query.",
  "The passage is on a related topic but does not supply what the query asks for.",
  "The passage partly supplies the information needed to answer or verify the query.",
  "The passage fully supplies the information needed to answer or verify the query.",
];
const TOP_LEVEL = RUBRIC.length - 1;

interface JevScoreAnswer {
  legend?: Record<string, string>;
  probabilities?: Record<string, number>;
}

interface JevResponse {
  answers?: Record<string, JevScoreAnswer>;
  usage?: { cost?: number };
}

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;
const TIMEOUT_MS = 20000;

function validateConfig(): void {
  if (!CLOUD_API.jev.apiKey) {
    throw new Error(
      "JEV_API_KEY environment variable is required when using Jev reranker provider " +
        "(on OpenRouter, OPENROUTER_API_KEY or the QWEN_API_KEY of the embeddings also works). " +
        "Set OSGREP_RERANK_PROVIDER=local to use local ColBERT reranking instead.",
    );
  }
}

function passageId(index: number): string {
  return `p${String(index + 1).padStart(2, "0")}`;
}

async function fetchWithRetry(
  query: string,
  documents: string[],
  attempt = 1,
): Promise<JevResponse> {
  const { apiKey, endpoint, model } = CLOUD_API.jev;

  const passages: Record<string, string> = {};
  const questions: Record<string, unknown> = {};
  documents.forEach((text, i) => {
    const id = passageId(i);
    passages[id] = text;
    questions[id] = {
      type: "score",
      instructions: `How well does passage ${id} supply the information needed to answer or verify the query?`,
      criteria: RUBRIC,
    };
  });

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        state: { query, passages },
        model,
        questions,
      }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });

    if (!response.ok) {
      const errorText = await response.text();

      // Handle rate limiting
      if (response.status === 429 && attempt < MAX_RETRIES) {
        const retryAfter =
          Number.parseInt(response.headers.get("retry-after") || "1", 10) *
          1000;
        const delay = Math.max(retryAfter, RETRY_DELAY_MS * attempt);
        log(`Jev: Rate limited, retrying in ${delay}ms (attempt ${attempt})`);
        await new Promise((resolve) => setTimeout(resolve, delay));
        return fetchWithRetry(query, documents, attempt + 1);
      }

      throw new Error(
        `Jev API error (${response.status}): ${errorText.slice(0, 300)}`,
      );
    }

    return (await response.json()) as JevResponse;
  } catch (error) {
    if (
      attempt < MAX_RETRIES &&
      error instanceof Error &&
      (error.message.includes("ECONNRESET") ||
        error.message.includes("ETIMEDOUT") ||
        error.message.includes("fetch failed"))
    ) {
      const delay = RETRY_DELAY_MS * attempt;
      log(`Jev: Network error, retrying in ${delay}ms (attempt ${attempt})`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return fetchWithRetry(query, documents, attempt + 1);
    }
    throw error;
  }
}

/**
 * Expected rubric level from the answer's distribution, scaled to 0..1
 * (0 = off-topic, 1 = fully supplies what the query asks for).
 */
function expectedLevel(id: string, answer: JevScoreAnswer | undefined): number {
  const legend = answer?.legend;
  const probabilities = answer?.probabilities;
  if (!legend || !probabilities) {
    throw new Error(`Jev reply has no readable answer for passage ${id}`);
  }

  let sum = 0;
  for (const [level, criterion] of Object.entries(legend)) {
    sum += (probabilities[level] ?? 0) * Math.max(0, RUBRIC.indexOf(criterion));
  }
  return sum / TOP_LEVEL;
}

/**
 * Rerank documents using Jev.
 *
 * @returns Scores array in the same order as input docs
 */
export async function jevRerank(
  input: RerankWithTextInput,
): Promise<RerankWithTextResult> {
  validateConfig();

  const documents = input.docs.map((d) => d.text);

  if (documents.length === 0) {
    return { scores: [] };
  }

  log(`Jev: Reranking ${documents.length} documents`);

  const started = Date.now();
  const response = await fetchWithRetry(input.query, documents);
  log(
    `Jev: Answered in ${Date.now() - started}ms, billed $${response.usage?.cost ?? "?"}`,
  );

  const scores = documents.map((_, i) => {
    const id = passageId(i);
    return expectedLevel(id, response.answers?.[id]);
  });

  return { scores };
}
