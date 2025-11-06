#!/usr/bin/env node
import { program } from "commander";
import * as fs from "fs";
import * as path from "path";
import { Mixedbread } from "@mixedbread/sdk";
import { isIgnoredByGit, getGitRepoFiles, computeBufferHash } from "./utils";

async function listStoreFileHashes(
  client: Mixedbread,
  store: string,
): Promise<Map<string, string | undefined>> {
  const byExternalId = new Map<string, string | undefined>();
  let after: string | null | undefined = undefined;
  do {
    const resp = await client.stores.files.list(store, { limit: 100, after });
    for (const f of resp.data) {
      const externalId = f.external_id ?? undefined;
      if (!externalId) continue;
      const metadata = (f.metadata as any) || {};
      const hash: string | undefined =
        typeof metadata?.hash === "string" ? metadata.hash : undefined;
      byExternalId.set(externalId, hash);
    }
    after = resp.pagination?.has_more
      ? (resp.pagination?.last_cursor ?? undefined)
      : undefined;
  } while (after);
  return byExternalId;
}

function filterRepoFiles(files: string[], repoRoot: string): string[] {
  const filtered: string[] = [];
  for (const filePath of files) {
    try {
      const stat = fs.statSync(filePath);
      if (!stat.isFile()) continue;
    } catch {
      continue;
    }
    if (isIgnoredByGit(filePath, repoRoot)) continue;
    filtered.push(filePath);
  }
  return filtered;
}

async function uploadFile(
  client: Mixedbread,
  store: string,
  filePath: string,
  fileName: string,
): Promise<void> {
  const buffer = fs.readFileSync(filePath);
  const hash = computeBufferHash(buffer);
  await client.stores.files.upload(
    store,
    new File([buffer], fileName, { type: "text/plain" }),
    {
      external_id: filePath,
      overwrite: true,
      metadata: {
        path: filePath,
        hash,
      },
    },
  );
}

async function initialSync(
  client: Mixedbread,
  store: string,
  repoRoot: string,
): Promise<void> {
  const storeHashes = await listStoreFileHashes(client, store);
  const repoFiles = filterRepoFiles(getGitRepoFiles(repoRoot), repoRoot);
  for (const filePath of repoFiles) {
    try {
      const buffer = fs.readFileSync(filePath);
      const hash = computeBufferHash(buffer);
      const existingHash = storeHashes.get(filePath);
      if (!existingHash || existingHash !== hash) {
        await uploadFile(client, store, filePath, path.basename(filePath));
        console.log(`Uploaded initial ${filePath}`);
      }
    } catch (err) {
      console.error("Failed to process initial file:", filePath, err);
    }
  }
}

program
  .version(
    JSON.parse(
      fs.readFileSync(path.join(__dirname, "../package.json"), {
        encoding: "utf-8",
      }),
    ).version,
  )
  .option("--api-key <string>", "The API key to use", process.env.MXBAI_API_KEY)
  .option(
    "--store <string>",
    "The store to use",
    process.env.MXBAI_STORE || "mgrep",
  );

program
  .command("search", { isDefault: true })
  .description("File pattern searcher")
  .argument("<pattern>", "The pattern to search for")
  .action(async (pattern, _options, cmd) => {
    const options: { apiKey: string; store: string } = cmd.optsWithGlobals();

    const mixedbread = new Mixedbread({
      apiKey: options.apiKey,
    });

    const results = await mixedbread.stores.search({
      query: pattern,
      store_identifiers: [options.store],
    });

    console.log(
      results.data
        .map((result) => {
          let content =
            result.type == "text"
              ? result.text
              : `Not a text chunk! (${result.type})`;
          content = JSON.stringify(content);
          return `${(result.metadata as any)?.path ?? "Unknown path"}: ${content}`;
        })
        .join("\n"),
    );
  });

program
  .command("watch")
  .description("Watch for file changes")
  .action(async (_args, cmd) => {
    const options: { apiKey: string; store: string } = cmd.optsWithGlobals();

    const mixedbread = new Mixedbread({
      apiKey: options.apiKey,
    });

    const watchRoot = process.cwd();
    console.log("Watching for file changes in", watchRoot);
    try {
      await initialSync(mixedbread, options.store, watchRoot);

      fs.watch(watchRoot, { recursive: true }, (eventType, rawFilename) => {
        const filename = rawFilename?.toString();
        if (!filename) {
          return;
        }
        const filePath = path.join(watchRoot, filename);
        console.log(`${eventType}: ${filePath}`);

        try {
          const stat = fs.statSync(filePath);
          if (!stat.isFile()) {
            return;
          }
        } catch {
          return;
        }

        if (isIgnoredByGit(filePath, watchRoot)) {
          return;
        }

        uploadFile(mixedbread, options.store, filePath, filename).catch(
          (err) => {
            console.error("Failed to upload changed file:", filePath, err);
          },
        );
      });
    } catch (err) {
      console.error("Failed to start watcher:", err);
      process.exitCode = 1;
    }
  });

program.parse();
