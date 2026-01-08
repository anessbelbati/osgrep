import * as os from "node:os";
import * as path from "node:path";

export const MODEL_IDS = {
  embed: "onnx-community/granite-embedding-30m-english-ONNX",
  colbert: "ryandono/mxbai-edge-colbert-v0-17m-onnx-int8",
};

// Provider types
export type EmbedProvider = "local" | "qwen";
export type RerankProvider = "local" | "zeroentropy";

// Provider configuration (read from env vars)
export const PROVIDERS = {
  embed: (process.env.OSGREP_EMBED_PROVIDER || "local") as EmbedProvider,
  rerank: (process.env.OSGREP_RERANK_PROVIDER || "local") as RerankProvider,
};

// Cloud API configuration
export const CLOUD_API = {
  qwen: {
    apiKey: process.env.QWEN_API_KEY || "",
    endpoint:
      process.env.QWEN_API_ENDPOINT ||
      "https://openrouter.ai/api/v1/embeddings",
    model: process.env.QWEN_MODEL || "qwen/qwen3-embedding-8b",
  },
  zeroentropy: {
    apiKey: process.env.ZEROENTROPY_API_KEY || "",
    endpoint:
      process.env.ZEROENTROPY_API_ENDPOINT ||
      "https://api.zeroentropy.dev/v1/models/rerank",
    model: process.env.ZEROENTROPY_MODEL || "zerank-2",
  },
};

const DEFAULT_WORKER_THREADS = (() => {
  const fromEnv = Number.parseInt(process.env.OSGREP_WORKER_THREADS ?? "", 10);
  if (Number.isFinite(fromEnv) && fromEnv > 0) return fromEnv;

  const cores = os.cpus().length || 1;
  // Higher cap for cloud providers (no local ONNX memory issues)
  const isCloud = process.env.OSGREP_EMBED_PROVIDER === "qwen";
  const HARD_CAP = isCloud ? 14 : 4;
  return Math.max(1, Math.min(HARD_CAP, isCloud ? 14 : cores));
})();

export const CONFIG = {
  VECTOR_DIM: PROVIDERS.embed === "qwen" ? 4096 : 384,
  COLBERT_DIM: 48,
  MAX_CHUNK_CHARS: 2000,
  MAX_CHUNK_LINES: 75,
  EMBED_BATCH_SIZE: 24,
  WORKER_THREADS: DEFAULT_WORKER_THREADS,
  QUERY_PREFIX: "",
};

export const WORKER_TIMEOUT_MS = Number.parseInt(
  process.env.OSGREP_WORKER_TIMEOUT_MS || "60000",
  10,
);

export const WORKER_BOOT_TIMEOUT_MS = Number.parseInt(
  process.env.OSGREP_WORKER_BOOT_TIMEOUT_MS || "300000",
  10,
);

export const MAX_WORKER_MEMORY_MB = Number.parseInt(
  process.env.OSGREP_MAX_WORKER_MEMORY_MB ||
  String(
    Math.max(
      2048,
      Math.floor((os.totalmem() / 1024 / 1024) * 0.5), // 50% of system RAM
    ),
  ),
  10,
);

const HOME = os.homedir();
const GLOBAL_ROOT = path.join(HOME, ".osgrep");

export const PATHS = {
  globalRoot: GLOBAL_ROOT,
  models: path.join(GLOBAL_ROOT, "models"),
  grammars: path.join(GLOBAL_ROOT, "grammars"),
};

export const MAX_FILE_SIZE_BYTES = 1024 * 1024 * 2; // 2MB limit for indexing

// Extensions we consider for indexing to avoid binary noise and improve relevance.
export const INDEXABLE_EXTENSIONS: Set<string> = new Set([
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".py",
  ".go",
  ".rs",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
  ".rb",
  ".php",
  ".cs",
  ".swift",
  ".kt",
  ".scala",
  ".lua",
  ".sh",
  ".sql",
  ".html",
  ".css",
  ".dart",
  ".el",
  ".clj",
  ".ex",
  ".exs",
  ".m",
  ".mm",
  ".f90",
  ".f95",
  ".json",
  ".yaml",
  ".yml",
  ".toml",
  ".xml",
  ".md",
  ".mdx",
  ".txt",

  ".gitignore",
  ".dockerfile",
  "dockerfile",
  "makefile",
]);
