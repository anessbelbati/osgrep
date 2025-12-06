---
name: osgrep
description: Semantic code search with call graph tracing for AI agents. Finds code by concept, understands structure (ORCHESTRATION vs DEFINITION), and maps dependencies. Use for "where is X", "how does Y work", "who calls Z", or understanding system architecture.
allowed-tools: "Bash(osgrep:*), Read"
license: Apache-2.0
---

# osgrep: Semantic Code Intelligence for AI Agents

## Two Core Commands

### 1. **Search** - Find code by concept
```bash
osgrep search "how does authentication work" --json
```
Returns semantically relevant code with roles, symbols, and context.

### 2. **Trace** - Understand relationships
```bash
osgrep search --trace "AuthService" --json
```
Returns call graph: who calls this (upstream) and what it calls (downstream).

---

## When to Use What

### Use `search` when:
- ✅ Finding implementations: *"where is validation logic"*
- ✅ Understanding concepts: *"how does middleware work"*
- ✅ Locating features: *"request body parsing"*
- ✅ Exploring architecture: *"authentication system"*

### Use `search --trace` when:
- ✅ Impact analysis: *"what depends on this function?"*
- ✅ Understanding flow: *"how does data reach this?"*
- ✅ Debugging: *"who calls this?"*
- ✅ Refactoring: *"what will break if I change this?"*

---

## Command Reference

### Search Command
```bash
osgrep search "query" [path] --json
```

**Options:**
- `--json` - **ALWAYS USE THIS**. Returns structured data instead of text.
- `[path]` - Optional: Scope search to directory (e.g., `src/auth/`)

**Example:**
```bash
osgrep search "middleware stack construction" --json
osgrep search "validation logic" src/api --json
```

### Trace Command
```bash
osgrep search --trace "SymbolName" --json
```

**Requirements:**
- Use **exact symbol name** (case-sensitive): `AuthService` not `auth service`
- Symbol must be indexed (recently defined in codebase)

**Example:**
```bash
osgrep search --trace "build_middleware_stack" --json
osgrep search --trace "solve_dependencies" --json
```

---

## JSON Output Format

### Search Results
```json
{
  "results": [
    {
      "text": "code snippet with breadcrumbs",
      "score": 0.85,
      "metadata": {
        "path": "src/auth.ts",
        "hash": "abc123..."
      },
      "generated_metadata": {
        "start_line": 42,
        "num_lines": 15,
        "type": "function"
      },
      "defined_symbols": ["login", "authenticate"],
      "referenced_symbols": ["jwt.sign", "db.users.find"],
      "role": "ORCHESTRATION",
      "parent_symbol": "AuthService",
      "context": ["// previous code context"]
    }
  ]
}
```

### Trace Results
```json
{
  "graph": {
    "center": {
      "symbol": "request_body_to_args",
      "file": "utils.py",
      "line": 487,
      "role": "ORCHESTRATION"
    },
    "callers": [
      {
        "symbol": "solve_dependencies",
        "file": "deps.py",
        "line": 716,
        "role": "ORCHESTRATION"
      }
    ],
    "callees": [
      "ModelField.validate",
      "FormData.items"
    ]
  },
  "metadata": {
    "count": 1,
    "query": "request_body_to_args"
  }
}
```

---

## How to Interpret Results

### Role Field (CRITICAL)
- **`ORCHESTRATION`** - Coordinates multiple operations. High complexity, many callees. **Start here for "how does X work"**.
- **`DEFINITION`** - Defines types, classes, functions. Low complexity. **Good for understanding structure**.

**Decision Rule:**
```
If role === "ORCHESTRATION" && score > 0.7:
  → This is likely the main logic. Read this file first.

If role === "DEFINITION":
  → Good for understanding types/interfaces, but may not show flow.
```

### Score Field
- **> 0.7** - Excellent match, very relevant
- **0.5-0.7** - Good match, likely relevant
- **< 0.5** - Tangential, consider only if no better options

### Symbols Arrays
- **`defined_symbols`** - What this code defines (functions, classes)
- **`referenced_symbols`** - What this code calls/uses
- **Use case:** Understanding dependencies without reading code

### Call Graph (Trace)
- **`callers`** - Upstream: who depends on this
- **`callees`** - Downstream: what this depends on
- **Count = importance:** High caller count = critical function

---

## Workflow Patterns

### Pattern 1: Understanding "How X Works"
```bash
# Step 1: Find the implementation
osgrep search "how does X work" --json

# Step 2: Look for ORCHESTRATION role with high score
results.filter(r => r.role === "ORCHESTRATION" && r.score > 0.7)

# Step 3: Trace it to see the flow
osgrep search --trace "main_function_from_step1" --json

# Step 4: Read key files
# Use file paths from steps 1-3
```

**Example:**
```bash
# User asks: "How does FastAPI validate requests?"

# Step 1: Search
osgrep search "request validation logic" --json
# → Returns get_request_handler (ORCHESTRATION, score 0.85)

# Step 2: Trace
osgrep search --trace "get_request_handler" --json
# → Shows callers: 8 route handlers
# → Shows callees: solve_dependencies, request_body_to_args

# Step 3: Read
Read routing.py:330 (get_request_handler)
Read dependencies/utils.py:487 (request_body_to_args)

# Response: "Validation starts at get_request_handler, which calls
# request_body_to_args to validate the body using Pydantic models."
```

### Pattern 2: Impact Analysis
```bash
# User asks: "What breaks if I change function X?"

# Step 1: Trace it
osgrep search --trace "function_name" --json

# Step 2: Check caller count
graph.callers.length  // 0 = safe, 10+ = high impact

# Step 3: Identify critical paths
graph.callers.filter(c => c.role === "ORCHESTRATION")

# Response based on count
```

### Pattern 3: Finding Definitions
```bash
# User asks: "Where is ClassName defined?"

# Option A: If you know the exact name
osgrep search --trace "ClassName" --json
# → center.file and center.line = definition location

# Option B: If you don't know exact name
osgrep search "ClassName definition" --json
# → Filter results by role === "DEFINITION"
```

### Pattern 4: Architecture Exploration
```bash
# User asks: "How does the auth system work?"

# Step 1: Broad search
osgrep search "authentication system architecture" --json

# Step 2: Survey results
# Look for multiple files in different directories
# Group by directory to see layers (controllers, services, middleware)

# Step 3: Trace key orchestrators
osgrep search --trace "main_auth_function" --json

# Step 4: Map the flow
# callers = entry points
# callees = what auth depends on
```

---

## Tips & Best Practices

### ✅ DO:
- **Always use `--json`** for structured, parseable output
- **Check `role` field first** - ORCHESTRATION = main logic
- **Use `score` to prioritize** - ignore results < 0.5
- **Combine search + trace** - Find symbols with search, trace for relationships
- **Read `breadcrumb` in text** - Shows context like `File: src/auth.ts > Class: AuthService > Function: login`
- **Look at caller/callee counts** - High counts = important functions

### ❌ DON'T:
- **Don't use for exact string matching** - Use `grep`/`rg` for that
- **Don't trace without searching first** - Trace requires exact symbol names
- **Don't ignore role field** - It's the most important metadata
- **Don't search for variable names** - Search for concepts, not identifiers
- **Don't assume low scores are wrong** - Sometimes the best match is 0.6

### Query Phrasing Guide
```bash
# ✅ GOOD - Conceptual, describes behavior
osgrep search "how background tasks are executed" --json
osgrep search "middleware stack construction logic" --json
osgrep search "request body validation" --json

# ❌ BAD - Too specific, like grep
osgrep search "BackgroundTasks.execute" --json
osgrep search "def build_middleware" --json

# ✅ GOOD - Architectural
osgrep search "authentication flow" --json
osgrep search "dependency injection system" --json

# ❌ BAD - Single words (too vague)
osgrep search "auth" --json
osgrep search "middleware" --json
```

---

## Troubleshooting

### No results / Low scores
- **Query too specific?** Try broader, conceptual phrases
- **Try related concepts:** "auth validation" → "request authentication"
- **Search for related symbols** then trace them

### Trace returns empty graph
- **Symbol not indexed?** Recently added code might not be indexed yet
- **Wrong symbol name?** Trace is case-sensitive and exact match
- **Try search first** to find the exact symbol name

### Results seem off-topic
- **Scope the search:** Add path argument: `osgrep search "query" src/auth/ --json`
- **Check scores:** Ignore results < 0.5
- **Filter by role:** Look for ORCHESTRATION results

### Need exact string match
- **Don't use osgrep** - Use `grep` or `rg` instead
- osgrep is for semantic/conceptual search, not exact strings

---

## Performance Notes

- **First search in a repo:** Automatically indexes (may take 10-60s)
- **Subsequent searches:** <100ms if `osgrep serve` is running
- **Trace queries:** Very fast (<50ms), just graph lookup
- **Large repos (100k+ files):** Index takes longer, but search stays fast

---

## Example Session

```bash
# User: "How does FastAPI's dependency injection work?"

# 1. Search for the concept
osgrep search "dependency injection system" --json
# → Result: solve_dependencies (ORCHESTRATION, score 0.82)

# 2. Trace to see the architecture
osgrep search --trace "solve_dependencies" --json
# → 16 callers (routing layer, websocket handlers)
# → 30 callees (request_params_to_args, request_body_to_args, etc.)

# 3. Understand the role
# ORCHESTRATION + 16 callers + 30 callees = CRITICAL ORCHESTRATOR

# 4. Read the file
Read dependencies/utils.py:628 (solve_dependencies function)

# Response:
"Dependency injection is handled by solve_dependencies() in
dependencies/utils.py. It's a critical orchestrator with 16 entry points
across the routing system. It resolves dependencies by calling
request_params_to_args for query/path params and request_body_to_args
for the request body, then recursively solves sub-dependencies."
```

---

## Summary

**osgrep gives you three superpowers:**
1. **Semantic search** - Find code by what it does, not what it's named
2. **Role understanding** - Know if code orchestrates (high-level) or defines (low-level)
3. **Call graph** - See relationships without reading code

**Use it whenever you need to understand unfamiliar code fast.**
