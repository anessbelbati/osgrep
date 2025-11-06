import { spawnSync } from "child_process";
import { createHash } from "crypto";
import * as path from "path";

export function computeBufferHash(buffer: Buffer): string {
  return createHash("sha256").update(buffer).digest("hex");
}

export function computeFileHash(
  filePath: string,
  readFileSyncFn: (p: string) => Buffer,
): string {
  const buffer = readFileSyncFn(filePath);
  return computeBufferHash(buffer);
}

export function getGitRepoFiles(repoRoot: string): string[] {
  const run = (args: string[]) => {
    const res = spawnSync("git", args, { cwd: repoRoot, encoding: "utf-8" });
    if (res.error) return "";
    return res.stdout as string;
  };

  // Tracked files
  const tracked = run(["ls-files", "-z"]).split("\u0000").filter(Boolean);

  // Untracked but not ignored
  const untracked = run(["ls-files", "--others", "--exclude-standard", "-z"])
    .split("\u0000")
    .filter(Boolean);

  const allRel = Array.from(new Set([...tracked, ...untracked]));
  return allRel.map((rel) => path.join(repoRoot, rel));
}

export function isIgnoredByGit(filePath: string, repoRoot: string): boolean {
  try {
    const result = spawnSync("git", ["check-ignore", "-q", "--", filePath], {
      cwd: repoRoot,
    });
    return result.status === 0;
  } catch {
    return false;
  }
}
