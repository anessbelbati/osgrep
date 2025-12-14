import { Command } from "commander";
import { GraphBuilder, type CallGraph } from "../lib/graph/graph-builder";
import { VectorDB } from "../lib/store/vector-db";
import { gracefulExit } from "../lib/utils/exit";
import { ensureProjectPaths, findProjectRoot } from "../lib/utils/project-root";

const style = {
  bold: (s: string) => `\x1b[1m${s}\x1b[22m`,
  dim: (s: string) => `\x1b[2m${s}\x1b[22m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[39m`,
  green: (s: string) => `\x1b[32m${s}\x1b[39m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[39m`,
};

/**
 * Format call graph for minimal agent-friendly output.
 * Format: file:line (space-separated for multiple)
 */
function formatPlain(graph: CallGraph, symbol: string): string {
  const lines: string[] = [];
  lines.push(symbol);

  if (graph.center) {
    lines.push(`  def: ${graph.center.file}:${graph.center.line}`);
  } else {
    lines.push("  def: (not found)");
  }

  if (graph.callees.length > 0) {
    lines.push(`  calls: ${graph.callees.join(" ")}`);
  }

  if (graph.callers.length > 0) {
    const callerLocs = graph.callers
      .map((c) => `${c.file}:${c.line}`)
      .join(" ");
    lines.push(`  called_by: ${callerLocs}`);
  }

  return lines.join("\n");
}

/**
 * Format call graph as a tree for human-readable output.
 */
function formatTree(graph: CallGraph, symbol: string): string {
  const lines: string[] = [];

  if (graph.center) {
    const role = graph.center.role ? ` ${style.dim(graph.center.role)}` : "";
    lines.push(
      `${style.bold(symbol)} (${style.cyan(`${graph.center.file}:${graph.center.line}`)})${role}`,
    );
  } else {
    lines.push(`${style.bold(symbol)} ${style.dim("(definition not found)")}`);
  }

  const hasCallees = graph.callees.length > 0;
  const hasCallers = graph.callers.length > 0;

  if (hasCallees) {
    const branch = hasCallers ? "\u251c\u2500\u2500" : "\u2514\u2500\u2500";
    lines.push(`${branch} ${style.yellow("calls:")}`);
    graph.callees.forEach((callee, i) => {
      const prefix = hasCallers ? "\u2502   " : "    ";
      const sym =
        i === graph.callees.length - 1
          ? "\u2514\u2500\u2500"
          : "\u251c\u2500\u2500";
      lines.push(`${prefix}${sym} ${callee}`);
    });
  }

  if (hasCallers) {
    lines.push(`\u2514\u2500\u2500 ${style.green("called by:")}`);
    graph.callers.forEach((caller, i) => {
      const sym =
        i === graph.callers.length - 1
          ? "\u2514\u2500\u2500"
          : "\u251c\u2500\u2500";
      lines.push(
        `    ${sym} ${caller.symbol} (${style.cyan(`${caller.file}:${caller.line}`)})`,
      );
    });
  }

  if (!hasCallees && !hasCallers) {
    lines.push(style.dim("  (no callers or callees found)"));
  }

  return lines.join("\n");
}

/**
 * Format call graph as JSON for programmatic use.
 */
function formatJson(graph: CallGraph, symbol: string): string {
  return JSON.stringify(
    {
      symbol,
      center: graph.center
        ? {
            file: graph.center.file,
            line: graph.center.line,
            role: graph.center.role,
          }
        : null,
      callers: graph.callers.map((c) => ({
        symbol: c.symbol,
        file: c.file,
        line: c.line,
      })),
      callees: graph.callees,
    },
    null,
    2,
  );
}

export const trace = new Command("trace")
  .description("Show call graph for a symbol (callers and callees)")
  .argument("<symbol>", "Symbol name to trace")
  .option("-d, --depth <n>", "Traversal depth (default: 1)", "1")
  .option("--callers", "Show only callers (who calls this)")
  .option("--callees", "Show only callees (what this calls)")
  .option("-p, --path <prefix>", "Filter to path prefix")
  .option("--pretty", "Pretty tree output (default for TTY)")
  .option("--plain", "Plain minimal output (default for non-TTY)")
  .option("--json", "JSON output")
  .action(async (symbol, cmd) => {
    const projectRoot = findProjectRoot(process.cwd()) ?? process.cwd();
    const paths = ensureProjectPaths(projectRoot);
    const db = new VectorDB(paths.lancedbDir);

    try {
      const builder = new GraphBuilder(db);

      const graph = await builder.buildGraph(symbol, {
        depth: Number.parseInt(cmd.depth, 10) || 1,
        callersOnly: cmd.callers as boolean,
        calleesOnly: cmd.callees as boolean,
        pathPrefix: cmd.path as string | undefined,
      });

      // Determine output format
      let output: string;
      if (cmd.json) {
        output = formatJson(graph, symbol);
      } else if (cmd.pretty) {
        output = formatTree(graph, symbol);
      } else if (cmd.plain || !process.stdout.isTTY) {
        output = formatPlain(graph, symbol);
      } else {
        output = formatTree(graph, symbol);
      }

      console.log(output);
    } finally {
      await db.close();
    }

    await gracefulExit();
  });
