import { workerManager } from "../workers/worker-manager";
import { CONFIG } from "../../config";
import { maxSim } from "./colbert-math";
import type {
    SearchFilter,
    SearchResponse,

    VectorRecord,
    ChunkType,
} from "../store/store";
import { VectorDB } from "../store/vector-db";

export class Searcher {
    // private readonly queryPrefix = CONFIG.QUERY_PREFIX;
    // private readonly colbertDim = CONFIG.COLBERT_DIM;

    constructor(private db: VectorDB) { }

    private mapRecordToChunk(record: Partial<VectorRecord>, score: number): ChunkType {
        const fullText = `${record.context_prev || ""}${record.content || ""}${record.context_next || ""} `;
        const startLine = record.start_line || 0;
        const endLine = record.end_line || startLine;

        return {
            type: "text",
            text: fullText,
            score,
            metadata: {
                path: record.path || "",
                hash: record.hash || "",
                is_anchor: !!record.is_anchor,
            },
            generated_metadata: {
                start_line: startLine,
                num_lines: Math.max(1, endLine - startLine + 1),
                type: record.chunk_type,
            },
        };
    }

    private applyStructureBoost(record: Partial<VectorRecord>, score: number): number {
        let adjusted = score;
        const chunkType = record.chunk_type || "";
        const boosted = ["function", "class", "method", "interface", "type_alias"];
        if (boosted.includes(chunkType)) {
            adjusted *= 1.25;
        }
        const pathStr = (record.path || "").toLowerCase();
        if (
            pathStr.includes(".test.") ||
            pathStr.includes(".spec.") ||
            pathStr.includes("__tests__")
        ) {
            adjusted *= 0.85;
        }
        if (
            pathStr.endsWith(".md") ||
            pathStr.endsWith(".mdx") ||
            pathStr.endsWith(".txt") ||
            pathStr.endsWith(".json") ||
            pathStr.endsWith(".lock") ||
            pathStr.includes("/docs/")
        ) {
            adjusted *= 0.5;
        }
        return adjusted;
    }

    async search(
        storeId: string,
        query: string,
        top_k?: number,
        _search_options?: { rerank?: boolean },
        _filters?: SearchFilter,
    ): Promise<SearchResponse> {
        const finalLimit = top_k ?? 10;

        // 1. Encode Query (Dense + ColBERT)
        const { dense: queryVector, colbert: queryMatrixRaw, colbertDim } =
            await workerManager.encodeQuery(query);

        if (colbertDim !== CONFIG.COLBERT_DIM) {
            console.warn(
                `[Searcher] Warning: Query ColBERT dim (${colbertDim}) != Config (${CONFIG.COLBERT_DIM})`,
            );
        }

        // Optimization: Convert query matrix to Float32Array[] once for simsimd
        const queryMatrix = queryMatrixRaw.map(v => new Float32Array(v));

        // 2. Fetch Candidates (Hybrid: Vector + FTS)
        // Optional path filter support
        const allFilters = Array.isArray((_filters as { all?: unknown })?.all)
            ? ((_filters as { all?: unknown }).all as Record<string, unknown>[])
            : [];
        const pathFilterEntry = allFilters.find(
            (f) => f?.key === "path" && f?.operator === "starts_with",
        );
        const pathPrefix =
            typeof pathFilterEntry?.value === "string" ? pathFilterEntry.value : "";

        const whereClause = pathPrefix
            ? `path LIKE '${pathPrefix.replace(/'/g, "''").replace(/\\/g, "\\\\")}%'`
            : undefined;

        // We fetch more candidates for reranking (e.g. 150 from each source)
        const PRE_RERANK_K = Math.max(finalLimit * 3, 150);

        let table;
        try {
            table = await this.db.ensureTable(storeId);
        } catch {
            return { data: [] };
        }

        let vectorQuery = table.vectorSearch(queryVector).limit(PRE_RERANK_K);
        if (whereClause) {
            vectorQuery = vectorQuery.where(whereClause);
        }
        const vectorResults = (await vectorQuery.toArray()) as VectorRecord[];

        let ftsResults: VectorRecord[] = [];
        try {
            let ftsQuery = table
                .search(query)
                .limit(PRE_RERANK_K);
            if (whereClause) {
                ftsQuery = ftsQuery.where(whereClause);
            }
            ftsResults = (await ftsQuery.toArray()) as VectorRecord[];
        } catch (e) {
            // FTS might fail if index doesn't exist or query is weird
            // console.warn("FTS search failed, falling back to vector only", e);
        }

        // Merge results (deduplicate by path + chunk_index)
        const seen = new Set<string>();
        const candidates: VectorRecord[] = [];

        for (const r of [...vectorResults, ...ftsResults]) {
            const key = `${r.path}:${r.chunk_index}`;
            if (!seen.has(key)) {
                seen.add(key);
                candidates.push(r);
            }
        }

        if (candidates.length === 0) {
            return { data: [] };
        }

        // 3. Rerank (ColBERT MaxSim)
        // We rerank ALL fetched candidates (up to ~300)
        const scored = candidates.map((doc) => {
            let score = 0;

            if (
                doc.colbert &&
                (Buffer.isBuffer(doc.colbert) || (doc.colbert as any) instanceof Uint8Array) &&
                queryMatrix.length > 0
            ) {
                const scale =
                    typeof doc.colbert_scale === "number" ? doc.colbert_scale : 1.0;
                let int8: Int8Array | null = null;
                if (Buffer.isBuffer(doc.colbert) || (doc.colbert as any) instanceof Uint8Array) {
                    const buffer = doc.colbert as Uint8Array;
                    int8 = new Int8Array(
                        buffer.buffer,
                        buffer.byteOffset,
                        buffer.byteLength,
                    );
                } else if (Array.isArray(doc.colbert)) {
                    int8 = new Int8Array(doc.colbert as number[]);
                }

                if (int8) {
                    // Reconstruct document matrix
                    // int8 is [seq_len * colbertDim]
                    // We know colbertDim (48)
                    // But we don't know seq_len explicitly, we derive it.
                    // Actually, we should use colbertDim from config or query?
                    // Ideally from config, but let's use the query's dim to match.
                    const dim = colbertDim;
                    const seqLen = int8.length / dim;

                    if (int8.length % dim !== 0) {
                        // console.warn(
                        //   `[Searcher] Chunk ${doc.path}:${doc.chunk_index} colbert buffer length ${int8.length} not divisible by ${dim}`,
                        // );
                    } else {
                        const docMatrix: Float32Array[] = [];
                        for (let i = 0; i < seqLen; i++) {
                            const start = i * dim;
                            const row = new Float32Array(dim);
                            for (let j = 0; j < dim; j++) {
                                row[j] = (int8[start + j] * scale) / 127.0;
                            }
                            docMatrix.push(row);
                        }

                        score = maxSim(queryMatrix, docMatrix);
                    }
                }
            } else {
                // Fallback score if no colbert data (shouldn't happen for indexed chunks)
                // Maybe use vector distance? But we don't have it easily here normalized.
                // Just 0.
            }

            return { record: doc, score };
        });

        // Apply structure boosting
        const boosted = scored.map((item) => {
            const boost = this.applyStructureBoost(item.record, item.score);
            return { ...item, score: boost };
        });

        // Sort by score descending
        boosted.sort((a, b) => b.score - a.score);

        // Take top K
        const finalResults = boosted.slice(0, finalLimit).map((item) => ({
            ...item.record,
            _score: item.score,
            // Remove heavy fields from response
            vector: undefined,
            colbert: undefined,
        }));

        return {
            data: finalResults.map((r) => this.mapRecordToChunk(r, r._score || 0)),
        };
    }
}
