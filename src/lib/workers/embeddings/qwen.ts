/**
 * Qwen Cloud Embedding Provider
 *
 * Uses the Qwen3-Embedding API via SiliconFlow (OpenAI-compatible endpoint).
 * Configure via environment variables:
 *   - QWEN_API_KEY: Your SiliconFlow API key
 *   - QWEN_API_ENDPOINT: Custom endpoint (optional)
 *   - QWEN_MODEL: Model name (optional, defaults to Qwen/Qwen3-Embedding-8B)
 */

import { CLOUD_API } from "../../../config";

const LOG_MODELS =
    process.env.OSGREP_DEBUG_MODELS === "1" ||
    process.env.OSGREP_DEBUG_MODELS === "true";
const log = (...args: unknown[]) => {
    if (LOG_MODELS) console.log(...args);
};

interface QwenEmbeddingResponse {
    object: string;
    data: Array<{
        object: string;
        index: number;
        embedding: number[];
    }>;
    model: string;
    usage: {
        prompt_tokens: number;
        total_tokens: number;
    };
}

export class QwenModel {
    private readonly maxRetries = 3;
    private readonly retryDelayMs = 1000;
    // Parallel API calls for faster embedding
    private readonly concurrency = Number.parseInt(
        process.env.QWEN_CONCURRENCY || "16",
        10,
    );
    private readonly batchSize = Number.parseInt(
        process.env.QWEN_BATCH_SIZE || "32",
        10,
    );

    private validateConfig(): void {
        if (!CLOUD_API.qwen.apiKey) {
            throw new Error(
                "QWEN_API_KEY environment variable is required when using Qwen embedding provider. " +
                "Set OSGREP_EMBED_PROVIDER=local to use local embeddings instead.",
            );
        }
    }

    isReady(): boolean {
        return !!CLOUD_API.qwen.apiKey;
    }

    async load(): Promise<void> {
        // Validate config on load - cloud providers don't need to load models
        this.validateConfig();
        log(`Qwen: Cloud provider ready (concurrency=${this.concurrency}, batchSize=${this.batchSize})`);
    }

    private async fetchWithRetry(
        texts: string[],
        attempt = 1,
    ): Promise<QwenEmbeddingResponse> {
        const { apiKey, endpoint, model } = CLOUD_API.qwen;

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${apiKey}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    model,
                    input: texts,
                    encoding_format: "float",
                }),
            });

            if (!response.ok) {
                const errorText = await response.text();

                // Handle rate limiting
                if (response.status === 429 && attempt < this.maxRetries) {
                    const retryAfter =
                        Number.parseInt(response.headers.get("retry-after") || "1", 10) *
                        1000;
                    const delay = Math.max(retryAfter, this.retryDelayMs * attempt);
                    log(`Qwen: Rate limited, retrying in ${delay}ms (attempt ${attempt})`);
                    await new Promise((resolve) => setTimeout(resolve, delay));
                    return this.fetchWithRetry(texts, attempt + 1);
                }

                throw new Error(
                    `Qwen API error (${response.status}): ${errorText}`,
                );
            }

            return (await response.json()) as QwenEmbeddingResponse;
        } catch (error) {
            if (
                attempt < this.maxRetries &&
                error instanceof Error &&
                (error.message.includes("ECONNRESET") ||
                    error.message.includes("ETIMEDOUT") ||
                    error.message.includes("fetch failed"))
            ) {
                const delay = this.retryDelayMs * attempt;
                log(`Qwen: Network error, retrying in ${delay}ms (attempt ${attempt})`);
                await new Promise((resolve) => setTimeout(resolve, delay));
                return this.fetchWithRetry(texts, attempt + 1);
            }
            throw error;
        }
    }

    private processResponse(response: QwenEmbeddingResponse): Float32Array[] {
        // Sort by index to ensure correct order
        const sortedData = response.data.sort((a, b) => a.index - b.index);

        return sortedData.map((item) => {
            const embedding = item.embedding;
            const result = new Float32Array(embedding.length);

            // Copy all dimensions
            for (let i = 0; i < embedding.length; i++) {
                result[i] = embedding[i];
            }

            // Normalize the vector
            let norm = 0;
            for (let i = 0; i < result.length; i++) {
                norm += result[i] * result[i];
            }
            norm = Math.sqrt(norm) || 1;
            for (let i = 0; i < result.length; i++) {
                result[i] /= norm;
            }

            return result;
        });
    }

    async runBatch(texts: string[]): Promise<Float32Array[]> {
        if (texts.length === 0) return [];

        this.validateConfig();

        // Split into batches
        const batches: string[][] = [];
        for (let i = 0; i < texts.length; i += this.batchSize) {
            batches.push(texts.slice(i, i + this.batchSize));
        }

        log(`Qwen: Embedding ${texts.length} texts in ${batches.length} batches (concurrency=${this.concurrency})`);

        // Process batches in parallel with concurrency limit
        const results: Float32Array[][] = [];
        for (let i = 0; i < batches.length; i += this.concurrency) {
            const chunk = batches.slice(i, i + this.concurrency);
            const chunkResults = await Promise.all(
                chunk.map(async (batch) => {
                    const response = await this.fetchWithRetry(batch);
                    return this.processResponse(response);
                }),
            );
            results.push(...chunkResults);
        }

        // Flatten results maintaining order
        return results.flat();
    }
}

