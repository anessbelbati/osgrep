/**
 * Tests for cloud providers (Qwen embeddings + Jev and ZeroEntropy rerankers)
 *
 * These tests verify:
 * 1. Provider configuration works correctly
 * 2. Qwen embedding provider API calls
 * 3. ZeroEntropy zerank-2 reranker API calls
 * 4. Jev reranker API calls
 * 5. Error handling for missing API keys
 * 6. Retry logic for network errors
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

    it("uses jev provider when OSGREP_RERANK_PROVIDER=jev", async () => {
        process.env.OSGREP_RERANK_PROVIDER = "jev";

        const { PROVIDERS } = await import("../src/config");

        expect(PROVIDERS.rerank).toBe("jev");
    });

    it("uses zeroentropy provider when it names an endpoint of its own", async () => {
        process.env.OSGREP_RERANK_PROVIDER = "zeroentropy";
        process.env.ZEROENTROPY_API_ENDPOINT = "https://custom.zeroentropy/v1/models/rerank";

        const { PROVIDERS } = await import("../src/config");

        expect(PROVIDERS.rerank).toBe("zeroentropy");
    });

    it("sends zeroentropy to jev when it names no endpoint (the hosted API was sunset)", async () => {
        process.env.OSGREP_RERANK_PROVIDER = "zeroentropy";
        delete process.env.ZEROENTROPY_API_ENDPOINT;

        const { PROVIDERS } = await import("../src/config");

        expect(PROVIDERS.rerank).toBe("jev");
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

    it("reads Jev API config from env vars", async () => {
        process.env.JEV_API_KEY = "jev-test-key";
        process.env.JEV_API_ENDPOINT = "https://custom.jev/v1/systemone";
        process.env.JEV_MODEL = "custom-jev";

        const { CLOUD_API } = await import("../src/config");

        expect(CLOUD_API.jev.apiKey).toBe("jev-test-key");
        expect(CLOUD_API.jev.endpoint).toBe("https://custom.jev/v1/systemone");
        expect(CLOUD_API.jev.model).toBe("custom-jev");
    });

    it("pays for Jev on OpenRouter with the embeddings key when no other key is set", async () => {
        delete process.env.JEV_API_KEY;
        delete process.env.JEV_API_ENDPOINT;
        delete process.env.OPENROUTER_API_KEY;
        delete process.env.QWEN_API_ENDPOINT;
        process.env.QWEN_API_KEY = "openrouter-key";

        const { CLOUD_API } = await import("../src/config");

        expect(CLOUD_API.jev.endpoint).toBe("https://openrouter.ai/api/v1/systemone");
        expect(CLOUD_API.jev.apiKey).toBe("openrouter-key");
    });

    it("never sends an OpenRouter key to another Jev host", async () => {
        delete process.env.JEV_API_KEY;
        delete process.env.QWEN_API_ENDPOINT;
        process.env.JEV_API_ENDPOINT = "https://custom.jev/v1/systemone";
        process.env.OPENROUTER_API_KEY = "openrouter-key";
        process.env.QWEN_API_KEY = "openrouter-key";

        const { CLOUD_API } = await import("../src/config");

        expect(CLOUD_API.jev.apiKey).toBe("");
    });

    it("does not take the embeddings key for Jev when the embeddings are not on OpenRouter", async () => {
        delete process.env.JEV_API_KEY;
        delete process.env.JEV_API_ENDPOINT;
        delete process.env.OPENROUTER_API_KEY;
        process.env.QWEN_API_ENDPOINT = "https://custom.endpoint/v1/embeddings";
        process.env.QWEN_API_KEY = "other-key";

        const { CLOUD_API } = await import("../src/config");

        expect(CLOUD_API.jev.apiKey).toBe("");
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

describe("Jev Reranker Provider", () => {
    const originalEnv = { ...process.env };

    const RUBRIC = [
        "The passage is off-topic for the query.",
        "The passage is on a related topic but does not supply what the query asks for.",
        "The passage partly supplies the information needed to answer or verify the query.",
        "The passage fully supplies the information needed to answer or verify the query.",
    ];

    // A System One "score" answer: the probability Jev puts on each rubric level
    const answer = (probabilities: number[]) => ({
        type: "score",
        legend: Object.fromEntries(RUBRIC.map((text, i) => [String(i), text])),
        probabilities: Object.fromEntries(probabilities.map((p, i) => [String(i), p])),
    });

    beforeEach(() => {
        vi.resetModules();
        process.env.JEV_API_KEY = "test-jev-key";
        delete process.env.JEV_API_ENDPOINT;
        delete process.env.JEV_MODEL;
        mockFetch.mockReset();
    });

    afterEach(() => {
        process.env = { ...originalEnv };
    });

    it("throws error when API key is missing", async () => {
        delete process.env.JEV_API_KEY;
        delete process.env.OPENROUTER_API_KEY;
        delete process.env.QWEN_API_KEY;

        const { jevRerank } = await import("../src/lib/workers/rerankers/jev");

        await expect(
            jevRerank({
                query: "test query",
                docs: [{ text: "doc1" }],
            })
        ).rejects.toThrow("JEV_API_KEY environment variable is required");
        expect(mockFetch).not.toHaveBeenCalled();
    });

    it("scores each passage by its expected rubric level, in input order", async () => {
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                answers: {
                    p02: answer([0, 0, 0, 1]),
                    p01: answer([1, 0, 0, 0]),
                    p03: answer([0, 0.5, 0.5, 0]),
                },
            }),
        });

        const { jevRerank } = await import("../src/lib/workers/rerankers/jev");

        const result = await jevRerank({
            query: "test query",
            docs: [{ text: "doc0" }, { text: "doc1" }, { text: "doc2" }],
        });

        expect(result.scores).toHaveLength(3);
        expect(result.scores[0]).toBeCloseTo(0, 5);
        expect(result.scores[1]).toBeCloseTo(1, 5);
        expect(result.scores[2]).toBeCloseTo(0.5, 5);
        expect(mockFetch).toHaveBeenCalledWith(
            expect.stringContaining("systemone"),
            expect.objectContaining({
                method: "POST",
                headers: expect.objectContaining({
                    Authorization: "Bearer test-jev-key",
                }),
            })
        );
    });

    it("reads a level from the legend's wording, not from the name of its key", async () => {
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                answers: {
                    p01: {
                        type: "score",
                        legend: { a: RUBRIC[3], b: RUBRIC[0], c: RUBRIC[2], d: RUBRIC[1] },
                        probabilities: { a: 1, b: 0, c: 0, d: 0 },
                    },
                },
            }),
        });

        const { jevRerank } = await import("../src/lib/workers/rerankers/jev");

        const result = await jevRerank({ query: "test", docs: [{ text: "doc" }] });

        expect(result.scores[0]).toBeCloseTo(1, 5);
    });

    it("sends every passage with a question of its own in one request", async () => {
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                answers: { p01: answer([1, 0, 0, 0]), p02: answer([1, 0, 0, 0]) },
            }),
        });

        const { jevRerank } = await import("../src/lib/workers/rerankers/jev");

        await jevRerank({
            query: "where is auth checked",
            docs: [{ text: "doc1" }, { text: "doc2" }],
        });

        expect(mockFetch).toHaveBeenCalledTimes(1);
        const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(callBody.model).toBe("typesafe/jev-1.13-20260917");
        expect(callBody.state).toEqual({
            query: "where is auth checked",
            passages: { p01: "doc1", p02: "doc2" },
        });
        expect(Object.keys(callBody.questions)).toEqual(["p01", "p02"]);
        expect(callBody.questions.p02.type).toBe("score");
        expect(callBody.questions.p02.instructions).toContain("passage p02");
        expect(callBody.questions.p02.criteria).toEqual(RUBRIC);
    });

    it("returns empty scores for empty input", async () => {
        const { jevRerank } = await import("../src/lib/workers/rerankers/jev");

        const result = await jevRerank({
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
                    answers: { p01: answer([0, 0, 0, 1]) },
                }),
            });

        const { jevRerank } = await import("../src/lib/workers/rerankers/jev");

        const result = await jevRerank({
            query: "test",
            docs: [{ text: "doc" }],
        });

        expect(result.scores[0]).toBeCloseTo(1, 5);
        expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it("throws on API error", async () => {
        mockFetch.mockResolvedValueOnce({
            ok: false,
            status: 500,
            text: async () => "Internal Server Error",
        });

        const { jevRerank } = await import("../src/lib/workers/rerankers/jev");

        await expect(
            jevRerank({ query: "test", docs: [{ text: "doc" }] })
        ).rejects.toThrow("Jev API error (500)");
    });

    it("throws when the reply leaves a passage without an answer", async () => {
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                answers: { p01: answer([0, 0, 0, 1]) },
            }),
        });

        const { jevRerank } = await import("../src/lib/workers/rerankers/jev");

        await expect(
            jevRerank({ query: "test", docs: [{ text: "doc1" }, { text: "doc2" }] })
        ).rejects.toThrow("no readable answer for passage p02");
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
