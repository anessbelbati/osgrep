# osgrep v3 Simplification Plan

## Current State (Post-Migration)

We've completed the major architectural change:
- **Rust core (osgrep-core)**: ONNX Runtime for dense + ColBERT embeddings
- **TypeScript orchestration**: syncer, watcher, chunker, searcher, LanceDB
- **Bridge**: `src/lib/native/index.ts` connects TS to Rust

**Results so far:**
- Code reduced from ~6,600 to ~4,400 lines (34% reduction)
- Binary size reduced from 58MB to 27MB (53% smaller)
- Removed: worker pool, skeleton, trace, graph modules

---

## Phase 1: Dependency Cleanup

### 1.1 Remove Dead Dependencies from package.json

**File:** `package.json`

Remove these dependencies that are no longer used:

```json
// REMOVE from dependencies:
"onnxruntime-node": "1.21.0",      // Now in Rust
"@huggingface/transformers": "^3.8.0",  // Now in Rust
"piscina": "^5.1.4",               // Worker pool deleted
```

**Why:** These were only used by the old TS embedding pipeline which is now handled by Rust.

### 1.2 Verify No Imports Remain

Search for any remaining imports of removed packages:
```bash
grep -r "onnxruntime" src/
grep -r "@huggingface/transformers" src/
grep -r "piscina" src/
```

---

## Phase 2: Simplify Search Pipeline

### 2.1 Delete Intent Detection

**Delete file:** `src/lib/search/intent.ts`

The intent detection system adds complexity without clear value:
- Regex-based query classification ("where is" → DEFINITION, "how does" → FLOW)
- Intent-based boosting in search results
- Not actually improving search quality in practice

### 2.2 Simplify Searcher

**File:** `src/lib/search/searcher.ts`

**Current complexity to remove:**

1. **Intent-based boosting** (lines 186-196):
   ```typescript
   // DELETE THIS BLOCK
   if (intent) {
     if (intent.type === "DEFINITION" && record.role === "DEFINITION") {
       boostFactor *= 1.2;
     }
     // ... more intent conditions
   }
   ```

2. **Role-based boosting** (lines 165-180):
   ```typescript
   // SIMPLIFY: Remove or reduce this complexity
   if (record.role === "ORCHESTRATION") {
     boostFactor *= 1.5;
   } else if (record.role === "DEFINITION") {
     boostFactor *= 1.2;
   }
   // ...
   ```

3. **Reference count boosting** (lines 174-180):
   ```typescript
   // DELETE: Over-engineered
   if (refs > 5) boostFactor *= 1.1;
   if (refs > 15) boostFactor *= 1.25;
   ```

**Simplified `applyStructureBoost` function:**

```typescript
private applyStructureBoost(
  record: Partial<VectorRecord>,
  score: number,
): number {
  let adjusted = score;

  // Anchor penalty (anchors are recall helpers, not results)
  if (record.is_anchor) {
    adjusted *= 0.99;
  }

  // Test file penalty
  const pathStr = (record.path || "").toLowerCase();
  const isTestPath =
    /(^|\/)(__tests__|tests?|specs?|benchmark)(\/|$)/i.test(pathStr) ||
    /\.(test|spec)\.[cm]?[jt]sx?$/i.test(pathStr);
  if (isTestPath) {
    adjusted *= 0.5;
  }

  // Docs/config penalty
  if (
    pathStr.endsWith(".md") ||
    pathStr.endsWith(".json") ||
    pathStr.endsWith(".lock") ||
    pathStr.includes("/docs/")
  ) {
    adjusted *= 0.6;
  }

  return adjusted;
}
```

### 2.3 Remove Intent from Search Signature

**Current:**
```typescript
async search(
  query: string,
  top_k?: number,
  _search_options?: { rerank?: boolean },
  _filters?: SearchFilter,
  pathPrefix?: string,
  intent?: SearchIntent,  // REMOVE
  signal?: AbortSignal,
): Promise<SearchResponse>
```

**After:**
```typescript
async search(
  query: string,
  top_k?: number,
  options?: { rerank?: boolean },
  filters?: SearchFilter,
  pathPrefix?: string,
  signal?: AbortSignal,
): Promise<SearchResponse>
```

### 2.4 Update Search Command

**File:** `src/commands/search.ts`

Remove any intent detection calls and simplify the search invocation.

---

## Phase 3: Code Quality Cleanup

### 3.1 Remove Environment Variable Overrides

The searcher has many env var overrides that add cognitive load:

```typescript
// Consider removing or documenting these:
OSGREP_ANCHOR_PENALTY
OSGREP_TEST_PENALTY
OSGREP_DOC_PENALTY
OSGREP_PRE_K
OSGREP_STAGE1_K
OSGREP_STAGE2_K
OSGREP_RERANK_TOP
OSGREP_RERANK_BLEND
OSGREP_MAX_PER_FILE
```

**Decision:** Either:
1. Remove all env vars and use fixed reasonable defaults
2. Keep only 2-3 most useful ones (e.g., `OSGREP_TOP_K`, `OSGREP_RERANK`)

### 3.2 Audit Remaining Files

Check for dead code in:
- `src/lib/store/` - vector-db.ts, types.ts
- `src/lib/core/` - chunker, languages
- `src/lib/utils/` - filter-builder, etc.

### 3.3 Type Cleanup

Remove unused types from `src/lib/store/types.ts` related to deleted features.

---

## Phase 4: Testing & Validation

### 4.1 Manual Testing

```bash
# Clean slate
rm -rf ~/.osgrep/indices/*

# Index a test repo
bun src/index.ts index --path . --reset

# Search queries
bun src/index.ts "how does embedding work"
bun src/index.ts "where is the config"
bun src/index.ts "vector search implementation"
```

### 4.2 Run Existing Tests

```bash
bun test
```

### 4.3 Typecheck

```bash
bunx tsc --noEmit
```

---

## Phase 5: Optional Further Simplifications

### 5.1 Consider Removing Two-Stage Rerank

The current pipeline has:
1. Vector search → top 500
2. RRF fusion (vector + FTS)
3. Stage 1: Pooled cosine filter → top 200
4. Stage 2: ColBERT MaxSim rerank → top 40
5. Structure boost + diversification

**Simpler alternative:**
1. Vector search → top 100
2. ColBERT rerank → top 20
3. Basic penalties (test/docs)

### 5.2 Consider Removing FTS Fallback

The FTS (full-text search) adds code but may not improve results significantly. Consider making it optional or removing entirely.

### 5.3 Simplify Output Formatting

`src/lib/output/formatter.ts` has complex role coloring and breadcrumb logic. Could be simplified to just show:
- File path + line number
- Code snippet
- Score

---

## File Inventory (Post-Simplification)

### Keep (Core)
```
src/
├── index.ts                    # CLI entry
├── config.ts                   # Constants
├── commands/
│   ├── index.ts               # Index command
│   ├── search.ts              # Search command
│   └── serve.ts               # MCP server
├── lib/
│   ├── native/
│   │   └── index.ts           # Rust bridge
│   ├── index/
│   │   ├── syncer.ts          # File sync
│   │   └── watcher.ts         # File watcher
│   ├── search/
│   │   └── searcher.ts        # Search logic (simplified)
│   ├── store/
│   │   ├── vector-db.ts       # LanceDB wrapper
│   │   └── types.ts           # Types
│   ├── core/
│   │   ├── chunker.ts         # Tree-sitter chunking
│   │   └── languages.ts       # Language configs
│   ├── output/
│   │   ├── formatter.ts       # Human output
│   │   └── json-formatter.ts  # JSON output
│   ├── utils/
│   │   └── ...                # Utilities
│   └── workers/
│       └── orchestrator.ts    # Embedding orchestrator
```

### Delete (This Phase)
```
src/lib/search/intent.ts       # Intent detection
```

### Already Deleted (Previous Phase)
```
src/lib/skeleton/              # Skeleton feature
src/lib/graph/                 # Call graph
src/lib/workers/pool.ts        # Worker pool
src/lib/workers/embeddings/    # TS embedding code
src/commands/skeleton.ts       # Skeleton command
src/commands/trace.ts          # Trace command
```

---

## Execution Checklist

- [ ] Phase 1.1: Remove dead deps from package.json
- [ ] Phase 1.2: Verify no imports remain
- [ ] Phase 2.1: Delete intent.ts
- [ ] Phase 2.2: Simplify applyStructureBoost
- [ ] Phase 2.3: Remove intent from search signature
- [ ] Phase 2.4: Update search command
- [ ] Phase 3.1: Decide on env var strategy
- [ ] Phase 4.1: Manual testing
- [ ] Phase 4.2: Run tests
- [ ] Phase 4.3: Typecheck
- [ ] Phase 5: Evaluate optional simplifications

---

## Success Criteria

1. **Code size:** < 4,000 lines of TypeScript
2. **Dependencies:** Remove 3 unused packages
3. **Complexity:** No intent detection, simplified boosting
4. **Functionality:** Index and search work correctly
5. **Maintainability:** Easy to understand search pipeline
