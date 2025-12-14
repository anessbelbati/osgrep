import { execSync } from "child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { join, extname } from "path";

/**
 * Clone a git repository if it doesn't exist locally
 */
export function cloneRepoIfMissing(repoUrl: string, dir: string): void {
  if (existsSync(dir)) {
    console.log(`Repository already exists at ${dir}`);
    return;
  }
  console.log(`Cloning ${repoUrl} to ${dir}...`);
  execSync(`git clone --depth 1 ${repoUrl} ${dir}`, { stdio: "inherit" });
}

/**
 * Recursively walk a directory and return all file paths
 * Ignores .git, node_modules, dist, target directories
 */
export function walkFiles(dir: string): string[] {
  const ignoreDirs = new Set([".git", "node_modules", "dist", "target", ".next", "build", "__pycache__"]);
  const results: string[] = [];

  function walk(currentDir: string): void {
    const entries = readdirSync(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = join(currentDir, entry.name);

      if (entry.isDirectory()) {
        if (!ignoreDirs.has(entry.name)) {
          walk(fullPath);
        }
      } else if (entry.isFile()) {
        results.push(fullPath);
      }
    }
  }

  walk(dir);
  return results;
}

/**
 * Check if a file has a code-like extension
 */
export function isCodeFile(filePath: string): boolean {
  const codeExtensions = new Set([
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".scala", ".cs",
    ".vue", ".svelte", ".astro",
    ".json", ".yaml", ".yml", ".toml",
    ".md", ".mdx", ".txt",
    ".sh", ".bash", ".zsh",
    ".sql", ".graphql", ".prisma",
    ".css", ".scss", ".less",
    ".html", ".xml"
  ]);
  return codeExtensions.has(extname(filePath).toLowerCase());
}

/**
 * Chunk file content by lines
 */
export function chunkFileByLines(text: string, linesPerChunk: number = 80): string[] {
  const lines = text.split("\n");
  const chunks: string[] = [];

  for (let i = 0; i < lines.length; i += linesPerChunk) {
    const chunk = lines.slice(i, i + linesPerChunk).join("\n");
    if (chunk.trim().length > 0) {
      chunks.push(chunk);
    }
  }

  return chunks;
}

/**
 * Read a file and return its contents, handling errors gracefully
 */
export function readFileSafe(filePath: string): string | null {
  try {
    const stat = statSync(filePath);
    // Skip files larger than 1MB
    if (stat.size > 1024 * 1024) {
      return null;
    }
    return readFileSync(filePath, "utf-8");
  } catch {
    return null;
  }
}

/**
 * Pick k candidates from chunks (for rerank benchmark)
 */
export function pickCandidates(chunks: string[], k: number): string[] {
  return chunks.slice(0, k);
}

/**
 * Load all code chunks from a repository
 */
export function loadRepoChunks(repoDir: string, linesPerChunk: number = 80): string[] {
  const files = walkFiles(repoDir);
  const codeFiles = files.filter(isCodeFile);
  const allChunks: string[] = [];

  for (const file of codeFiles) {
    const content = readFileSafe(file);
    if (content) {
      const chunks = chunkFileByLines(content, linesPerChunk);
      allChunks.push(...chunks);
    }
  }

  return allChunks;
}

/**
 * Chunk metadata for tracking file/line mapping
 */
export interface ChunkMeta {
  id: number;
  file: string;       // relative to repo root
  startLine: number;  // 1-indexed
  endLine: number;    // 1-indexed, inclusive
  text: string;
}

/**
 * Load all code chunks from a repository with metadata
 * Returns chunks with file/line info for ground truth matching
 *
 * @param repoDir - The repository directory path
 * @param linesPerChunk - Lines per chunk (default 80)
 * @param repoName - Optional repo name to prefix paths (for CodeAtlas compatibility)
 */
export function loadRepoChunksWithMeta(
  repoDir: string,
  linesPerChunk: number = 80,
  repoName?: string
): ChunkMeta[] {
  const files = walkFiles(repoDir);
  const codeFiles = files.filter(isCodeFile);
  const allChunks: ChunkMeta[] = [];
  let chunkId = 0;

  // Normalize repoDir - remove leading ./ and trailing / for consistent matching
  const normalizedRepoDir = repoDir.replace(/^\.\//, '').replace(/\/+$/, '');

  for (const file of codeFiles) {
    const content = readFileSafe(file);
    if (!content) continue;

    // Normalize file path too - remove leading ./
    const normalizedFile = file.replace(/^\.\//, '');

    // Get relative path from repo root
    let relativePath = normalizedFile.startsWith(normalizedRepoDir + '/')
      ? normalizedFile.slice(normalizedRepoDir.length + 1)
      : normalizedFile;

    // If repoName provided, prefix the path (for CodeAtlas format compatibility)
    // CodeAtlas uses paths like "aiohttp/web_protocol.py"
    if (repoName) {
      relativePath = `${repoName}/${relativePath}`;
    }

    const lines = content.split("\n");

    for (let i = 0; i < lines.length; i += linesPerChunk) {
      const chunkLines = lines.slice(i, i + linesPerChunk);
      const text = chunkLines.join("\n");

      if (text.trim().length > 0) {
        allChunks.push({
          id: chunkId++,
          file: relativePath,
          startLine: i + 1,  // 1-indexed
          endLine: Math.min(i + linesPerChunk, lines.length),  // 1-indexed, inclusive
          text,
        });
      }
    }
  }

  return allChunks;
}

/**
 * Check if a chunk overlaps with a positive span
 * Handles inconsistent CodeAtlas file path formats (some have repo prefix, some don't)
 */
export function chunkOverlapsSpan(
  chunk: ChunkMeta,
  span: { file: string; start_line: number; end_line: number }
): boolean {
  // Normalize paths - remove leading slashes
  const chunkFile = chunk.file.replace(/^\//, '');
  const spanFile = span.file.replace(/^\//, '');

  // Check file match:
  // 1. Exact match: "aiohttp/web_protocol.py" === "aiohttp/web_protocol.py"
  // 2. Chunk ends with span: "chi/mux.go" ends with "/mux.go" (for span "mux.go")
  // 3. Span ends with chunk: "mux.go" (span without prefix) matches end of "chi/mux.go"
  const filesMatch =
    chunkFile === spanFile ||
    chunkFile.endsWith('/' + spanFile) ||
    spanFile.endsWith('/' + chunkFile);

  if (!filesMatch) return false;

  // Check line overlap: chunk.startLine <= span.end_line AND chunk.endLine >= span.start_line
  return chunk.startLine <= span.end_line && chunk.endLine >= span.start_line;
}
