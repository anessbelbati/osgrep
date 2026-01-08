/**
 * ZeroEntropy Reranker Provider (zerank-2)
 *
 * Uses the ZeroEntropy API for document reranking.
 * Configure via environment variables:
 *   - ZEROENTROPY_API_KEY: Your ZeroEntropy API key
 *   - ZEROENTROPY_API_ENDPOINT: Custom endpoint (optional)
 *   - ZEROENTROPY_MODEL: Model name (optional, defaults to zerank-2)
 */

import { CLOUD_API } from "../../../config";

const LOG_MODELS =
    process.env.OSGREP_DEBUG_MODELS === "1" ||
    process.env.OSGREP_DEBUG_MODELS === "true";
const log = (...args: unknown[]) => {
    if (LOG_MODELS) console.log(...args);
};

interface ZerankRerankResult {
    index: number;
    relevance_score: number;
}

interface ZerankRerankResponse {
    model: string;
    results: ZerankRerankResult[];
}

export interface RerankWithTextInput {
    query: string;
    docs: Array<{
        text: string;
    }>;
    topN?: number;
}

export interface RerankWithTextResult {
    scores: number[];
}

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;

function validateConfig(): void {
    if (!CLOUD_API.zeroentropy.apiKey) {
        throw new Error(
            "ZEROENTROPY_API_KEY environment variable is required when using ZeroEntropy reranker provider. " +
            "Set OSGREP_RERANK_PROVIDER=local to use local ColBERT reranking instead.",
        );
    }
}

async function fetchWithRetry(
    query: string,
    documents: string[],
    topN: number,
    attempt = 1,
): Promise<ZerankRerankResponse> {
    const { apiKey, endpoint, model } = CLOUD_API.zeroentropy;

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${apiKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model,
                query,
                documents,
                top_n: topN,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();

            // Handle rate limiting
            if (response.status === 429 && attempt < MAX_RETRIES) {
                const retryAfter =
                    Number.parseInt(response.headers.get("retry-after") || "1", 10) *
                    1000;
                const delay = Math.max(retryAfter, RETRY_DELAY_MS * attempt);
                log(`ZeroEntropy: Rate limited, retrying in ${delay}ms (attempt ${attempt})`);
                await new Promise((resolve) => setTimeout(resolve, delay));
                return fetchWithRetry(query, documents, topN, attempt + 1);
            }

            throw new Error(`ZeroEntropy API error (${response.status}): ${errorText}`);
        }

        return (await response.json()) as ZerankRerankResponse;
    } catch (error) {
        if (
            attempt < MAX_RETRIES &&
            error instanceof Error &&
            (error.message.includes("ECONNRESET") ||
                error.message.includes("ETIMEDOUT") ||
                error.message.includes("fetch failed"))
        ) {
            const delay = RETRY_DELAY_MS * attempt;
            log(`ZeroEntropy: Network error, retrying in ${delay}ms (attempt ${attempt})`);
            await new Promise((resolve) => setTimeout(resolve, delay));
            return fetchWithRetry(query, documents, topN, attempt + 1);
        }
        throw error;
    }
}

/**
 * Rerank documents using ZeroEntropy's zerank-2 API.
 *
 * @returns Scores array in the same order as input docs
 */
export async function zerankRerank(
    input: RerankWithTextInput,
): Promise<RerankWithTextResult> {
    validateConfig();

    const { query, docs, topN } = input;
    const documents = docs.map((d) => d.text);

    if (documents.length === 0) {
        return { scores: [] };
    }

    log(`ZeroEntropy: Reranking ${documents.length} documents`);

    const response = await fetchWithRetry(
        query,
        documents,
        topN ?? documents.length,
    );

    // Initialize scores array with zeros for all docs
    const scores = new Array(documents.length).fill(0);

    // Map returned scores back to original indices
    for (const result of response.results) {
        if (result.index >= 0 && result.index < scores.length) {
            scores[result.index] = result.relevance_score;
        }
    }

    return { scores };
}
