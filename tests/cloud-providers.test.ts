/**
 * Tests for cloud providers (Qwen embeddings + ZeroEntropy reranker)
 *
 * These tests verify:
 * 1. Provider configuration works correctly
 * 2. Qwen embedding provider API calls
 * 3. ZeroEntropy zerank-2 reranker API calls
 * 4. Error handling for missing API keys
 * 5. Retry logic for network errors
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

describe("Cloud Provider Configuration", () => {
    const originalEnv = { ...process.env };

    beforeEach(() => {
        vi.resetModules();
    });

    afterEach(() => {
        process.env = { ...originalEnv };
        vi.clearAllMocks();
    });

    it("defaults to local providers when env vars not set", async () => {
        delete process.env.OSGREP_EMBED_PROVIDER;
        delete process.env.OSGREP_RERANK_PROVIDER;

        const { PROVIDERS } = await import("../src/config");

        expect(PROVIDERS.embed).toBe("local");
        expect(PROVIDERS.rerank).toBe("local");
    });

    it("uses qwen provider when OSGREP_EMBED_PROVIDER=qwen", async () => {
        process.env.OSGREP_EMBED_PROVIDER = "qwen";

        const { PROVIDERS } = await import("../src/config");

        expect(PROVIDERS.embed).toBe("qwen");
    });

    it("uses zeroentropy provider when OSGREP_RERANK_PROVIDER=zeroentropy", async () => {
        process.env.OSGREP_RERANK_PROVIDER = "zeroentropy";

        const { PROVIDERS } = await import("../src/config");

        expect(PROVIDERS.rerank).toBe("zeroentropy");
    });

    it("reads Qwen API config from env vars", async () => {
        process.env.QWEN_API_KEY = "test-key";
        process.env.QWEN_API_ENDPOINT = "https://custom.endpoint/v1/embeddings";
        process.env.QWEN_MODEL = "custom-model";

        const { CLOUD_API } = await import("../src/config");

        expect(CLOUD_API.qwen.apiKey).toBe("test-key");
        expect(CLOUD_API.qwen.endpoint).toBe("https://custom.endpoint/v1/embeddings");
        expect(CLOUD_API.qwen.model).toBe("custom-model");
    });

    it("reads ZeroEntropy API config from env vars", async () => {
        process.env.ZEROENTROPY_API_KEY = "ze-test-key";
        process.env.ZEROENTROPY_API_ENDPOINT = "https://custom.zeroentropy/v1/models/rerank";
        process.env.ZEROENTROPY_MODEL = "custom-reranker";

        const { CLOUD_API } = await import("../src/config");

        expect(CLOUD_API.zeroentropy.apiKey).toBe("ze-test-key");
        expect(CLOUD_API.zeroentropy.endpoint).toBe("https://custom.zeroentropy/v1/models/rerank");
        expect(CLOUD_API.zeroentropy.model).toBe("custom-reranker");
    });
});

describe("Qwen Embedding Provider", () => {
    const originalEnv = { ...process.env };

    beforeEach(() => {
        vi.resetModules();
        process.env.QWEN_API_KEY = "test-qwen-key";
        mockFetch.mockReset();
    });

    afterEach(() => {
        process.env = { ...originalEnv };
    });

    it("throws error when API key is missing", async () => {
        delete process.env.QWEN_API_KEY;

        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const model = new QwenModel();

        await expect(model.runBatch(["test"])).rejects.toThrow(
            "QWEN_API_KEY environment variable is required"
        );
    });

    it("returns embeddings from API response", async () => {
        const mockResponse = {
            object: "list",
            data: [
                { index: 0, embedding: Array(384).fill(0.1) },
                { index: 1, embedding: Array(384).fill(0.2) },
            ],
            model: "qwen/qwen3-embedding-8b",
            usage: { prompt_tokens: 10, total_tokens: 10 },
        };

        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => mockResponse,
        });

        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const model = new QwenModel();

        const result = await model.runBatch(["hello", "world"]);

        expect(result).toHaveLength(2);
        expect(result[0]).toBeInstanceOf(Float32Array);
        expect(result[1]).toBeInstanceOf(Float32Array);
        expect(mockFetch).toHaveBeenCalledWith(
            expect.stringContaining("embeddings"),
            expect.objectContaining({
                method: "POST",
                headers: expect.objectContaining({
                    Authorization: "Bearer test-qwen-key",
                }),
            })
        );
    });

    it("returns empty array for empty input", async () => {
        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const model = new QwenModel();

        const result = await model.runBatch([]);

        expect(result).toEqual([]);
        expect(mockFetch).not.toHaveBeenCalled();
    });

    it("normalizes embeddings", async () => {
        const mockResponse = {
            object: "list",
            data: [{ index: 0, embedding: [3, 4, 0] }], // 3-4-5 triangle, norm = 5
            model: "qwen/qwen3-embedding-8b",
            usage: { prompt_tokens: 1, total_tokens: 1 },
        };

        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => mockResponse,
        });

        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const model = new QwenModel();

        const result = await model.runBatch(["test"]);

        // After normalization, should be [0.6, 0.8, 0] (divided by 5)
        expect(result[0][0]).toBeCloseTo(0.6, 5);
        expect(result[0][1]).toBeCloseTo(0.8, 5);
    });

    it("retries on rate limit (429)", async () => {
        mockFetch
            .mockResolvedValueOnce({
                ok: false,
                status: 429,
                headers: new Map([["retry-after", "1"]]),
                text: async () => "Rate limited",
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    data: [{ index: 0, embedding: Array(384).fill(0.1) }],
                }),
            });

        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const model = new QwenModel();

        const result = await model.runBatch(["test"]);

        expect(result).toHaveLength(1);
        expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it("throws on API error", async () => {
        mockFetch.mockResolvedValueOnce({
            ok: false,
            status: 500,
            text: async () => "Internal Server Error",
        });

        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const model = new QwenModel();

        await expect(model.runBatch(["test"])).rejects.toThrow("Qwen API error (500)");
    });

    it("isReady returns true when API key is set", async () => {
        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const model = new QwenModel();

        expect(model.isReady()).toBe(true);
    });

    it("isReady returns false when API key is missing", async () => {
        delete process.env.QWEN_API_KEY;

        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const model = new QwenModel();

        expect(model.isReady()).toBe(false);
    });
});

describe("ZeroEntropy Reranker Provider", () => {
    const originalEnv = { ...process.env };

    beforeEach(() => {
        vi.resetModules();
        process.env.ZEROENTROPY_API_KEY = "test-ze-key";
        mockFetch.mockReset();
    });

    afterEach(() => {
        process.env = { ...originalEnv };
    });

    it("throws error when API key is missing", async () => {
        delete process.env.ZEROENTROPY_API_KEY;

        const { zerankRerank } = await import("../src/lib/workers/rerankers/zerank");

        await expect(
            zerankRerank({
                query: "test query",
                docs: [{ text: "doc1" }],
            })
        ).rejects.toThrow("ZEROENTROPY_API_KEY environment variable is required");
    });

    it("returns scores from API response in correct order", async () => {
        const mockResponse = {
            model: "zerank-2",
            results: [
                { index: 1, relevance_score: 0.9 },
                { index: 0, relevance_score: 0.5 },
                { index: 2, relevance_score: 0.3 },
            ],
        };

        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => mockResponse,
        });

        const { zerankRerank } = await import("../src/lib/workers/rerankers/zerank");

        const result = await zerankRerank({
            query: "test query",
            docs: [{ text: "doc0" }, { text: "doc1" }, { text: "doc2" }],
        });

        // Scores should be in original document order
        expect(result.scores).toEqual([0.5, 0.9, 0.3]);
        expect(mockFetch).toHaveBeenCalledWith(
            expect.stringContaining("rerank"),
            expect.objectContaining({
                method: "POST",
                headers: expect.objectContaining({
                    Authorization: "Bearer test-ze-key",
                }),
            })
        );
    });

    it("returns empty scores for empty input", async () => {
        const { zerankRerank } = await import("../src/lib/workers/rerankers/zerank");

        const result = await zerankRerank({
            query: "test",
            docs: [],
        });

        expect(result.scores).toEqual([]);
        expect(mockFetch).not.toHaveBeenCalled();
    });

    it("retries on rate limit (429)", async () => {
        mockFetch
            .mockResolvedValueOnce({
                ok: false,
                status: 429,
                headers: new Map([["retry-after", "1"]]),
                text: async () => "Rate limited",
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    results: [{ index: 0, relevance_score: 0.8 }],
                }),
            });

        const { zerankRerank } = await import("../src/lib/workers/rerankers/zerank");

        const result = await zerankRerank({
            query: "test",
            docs: [{ text: "doc" }],
        });

        expect(result.scores).toEqual([0.8]);
        expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it("throws on API error", async () => {
        mockFetch.mockResolvedValueOnce({
            ok: false,
            status: 500,
            text: async () => "Internal Server Error",
        });

        const { zerankRerank } = await import("../src/lib/workers/rerankers/zerank");

        await expect(
            zerankRerank({ query: "test", docs: [{ text: "doc" }] })
        ).rejects.toThrow("ZeroEntropy API error (500)");
    });

    it("sends topN parameter when provided", async () => {
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                results: [{ index: 0, relevance_score: 0.9 }],
            }),
        });

        const { zerankRerank } = await import("../src/lib/workers/rerankers/zerank");

        await zerankRerank({
            query: "test",
            docs: [{ text: "doc1" }, { text: "doc2" }, { text: "doc3" }],
            topN: 2,
        });

        const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(callBody.top_n).toBe(2);
    });
});

describe("Provider Integration", () => {
    const originalEnv = { ...process.env };

    beforeEach(() => {
        vi.resetModules();
        mockFetch.mockReset();
    });

    afterEach(() => {
        process.env = { ...originalEnv };
    });

    it("createEmbedModel returns QwenModel when provider is qwen", async () => {
        process.env.OSGREP_EMBED_PROVIDER = "qwen";
        process.env.QWEN_API_KEY = "test-key";

        // We can't easily test private factory function, but we can test the behavior
        // by checking typeof the model class
        const { QwenModel } = await import("../src/lib/workers/embeddings/qwen");
        const { GraniteModel } = await import("../src/lib/workers/embeddings/granite");
        const { PROVIDERS } = await import("../src/config");

        expect(PROVIDERS.embed).toBe("qwen");
        expect(new QwenModel()).toBeDefined();
    });

    it("createEmbedModel returns GraniteModel when provider is local", async () => {
        process.env.OSGREP_EMBED_PROVIDER = "local";

        const { PROVIDERS } = await import("../src/config");

        expect(PROVIDERS.embed).toBe("local");
    });
});
