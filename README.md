# ctxgraph

**Local-first context graph engine for AI agents and human teams.**

Zero infrastructure. Zero API cost. Single Rust binary.

---

ctxgraph stores decision traces — the full story behind every choice — and makes them searchable. It runs entirely on your machine. No Neo4j. No OpenAI API key. No Docker. No Python. One binary, one SQLite file, instant startup.

When someone (or an AI agent) asks *"why did we do X?"*, ctxgraph traverses the graph of past decisions, finds relevant precedents, and returns the full context — who decided, when, what alternatives were considered, which policies applied, and what the outcome was.

## Why ctxgraph

Every existing context graph implementation requires heavy infrastructure:

| | Graphiti (Zep) | ctxgraph |
|---|---|---|
| Graph database | Neo4j / FalkorDB (Docker) | SQLite (embedded) |
| LLM API key | Required (OpenAI/Anthropic) | Not required |
| Runtime | Python 3.10+ | Single Rust binary |
| Cost per episode | ~$0.01-0.05 (LLM tokens) | $0.00 |
| Setup time | 15-30 minutes | 5 seconds |
| Internet required | Yes (always) | No (fully offline) |
| Privacy | Data sent to OpenAI | Nothing leaves your machine |

ctxgraph is for the 90% of use cases where you don't need Neo4j-scale infrastructure.

## Quick Start

```bash
# Initialize in your project
ctxgraph init

# Log decisions
ctxgraph log "Chose Postgres over SQLite for billing. Reason: concurrent writes."
ctxgraph log --source slack "Priya approved the discount for Reliance"
ctxgraph log --tags "architecture,database" "Switched from REST to gRPC"

# Search
ctxgraph query "why Postgres?"
ctxgraph query "discount precedents" --limit 5

# Auto-capture git commits (v0.4+)
ctxgraph watch --git --last 50

# Stats
ctxgraph stats
```

## How It Works

```
Your App / CLI / AI Agent
         |
    ctxgraph engine
         |
    ┌─────────────────────────────────┐
    │  Extraction Pipeline            │
    │                                 │
    │  Tier 1: GLiNER2 (ONNX)        │  $0, 2-10ms, local
    │    entities + relations         │
    │    + temporal heuristics        │
    │                                 │
    │  Tier 2: Coreference + Dedup   │  $0, 15-50ms, local
    │                                 │
    │  Tier 3: Ollama / LLM API      │  optional, opt-in
    └─────────────────────────────────┘
         |
    ┌─────────────────────────────────┐
    │  Storage: SQLite + FTS5         │
    │  Bi-temporal timestamps         │
    │  Graph via recursive CTEs       │
    └─────────────────────────────────┘
         |
    ┌─────────────────────────────────┐
    │  Query: FTS5 + Semantic + Graph │
    │  Fused via Reciprocal Rank      │
    │  MCP server for AI agents       │
    └─────────────────────────────────┘
```

### Three-Tier Extraction

ctxgraph uses a tiered extraction system. Each tier adds capability and cost. Tier 1 is always available at zero cost.

**Tier 1 — Schema-Driven Local** (always on, $0, 2-10ms)
- GLiNER2 via ONNX Runtime — zero-shot entity + relation extraction on CPU
- User-defined labels in `ctxgraph.toml` (Person, Component, Decision, Reason, etc.)
- Temporal heuristics for date parsing
- ~85% accuracy on semi-structured text

**Tier 2 — Enhanced Local** (default on, $0, 15-50ms)
- Coreference resolution ("she" → nearest Person entity)
- Fuzzy entity dedup via Jaro-Winkler ("P. Sharma" = "Priya Sharma")
- Context-aware temporal resolution
- ~90% accuracy on semi-structured text

**Tier 3 — LLM-Enhanced** (opt-in, $0 with Ollama)
- Contradiction detection between old and new facts
- Complex temporal reasoning
- Community summarization
- Unstructured text extraction
- ~93-95% accuracy

### Bi-Temporal Model

Every relationship has two time dimensions:

- **valid_from / valid_until** — When was this fact true in the real world?
- **recorded_at** — When was this fact recorded in ctxgraph?

Facts are never deleted — they are invalidated. The full history is preserved.

```
Edge: Alice →[works_at]→ Google
  valid_from:  2020-01-15
  valid_until: 2025-06-01     ← invalidated when she joined Meta

Edge: Alice →[works_at]→ Meta
  valid_from:  2025-06-01
  valid_until: null            ← currently true
```

### Search

Three modes fused via Reciprocal Rank Fusion:

- **FTS5** — keyword matching across episodes, entities, edges
- **Semantic** — 384-dim embeddings via all-MiniLM-L6-v2 (local, ONNX)
- **Graph traversal** — multi-hop walk via recursive CTEs

A result appearing in multiple modes is ranked highest.

## MCP Server

ctxgraph runs as an MCP server for AI agents (Claude Desktop, Cursor, Claude Code):

```json
{
  "mcpServers": {
    "ctxgraph": {
      "command": "ctxgraph",
      "args": ["mcp", "start"]
    }
  }
}
```

Exposes 5 tools:

| Tool | Description |
|---|---|
| `ctxgraph_add_episode` | Record a decision or event |
| `ctxgraph_search` | Search for relevant decisions and precedents |
| `ctxgraph_get_decision` | Get full decision trace by ID |
| `ctxgraph_traverse` | Walk the graph from an entity |
| `ctxgraph_find_precedents` | Find similar past decisions |

## CLI Reference

```
ctxgraph init [--name <name>]                Initialize .ctxgraph/ in current directory
ctxgraph log <text> [--source <src>] [--tags <t1,t2>]   Log a decision or event
ctxgraph query <text> [--limit <n>]          Search the context graph
ctxgraph entities list [--type <type>]       List entities
ctxgraph entities show <id>                  Show entity with relationships
ctxgraph decisions list                      List episodes
ctxgraph decisions show <id>                 Show full decision trace
ctxgraph stats                               Graph statistics
ctxgraph watch --git [--last <n>]            Auto-capture git commits
ctxgraph models download                     Download ONNX models
ctxgraph export --format json|csv            Export graph data
ctxgraph mcp start                           Run as MCP server
```

## Configuration

```toml
# ctxgraph.toml

[schema]
name = "default"

[schema.entities]
Person = "A person involved in a decision"
Component = "A software component or technology"
Service = "A service or application"
Decision = "An explicit choice that was made"
Reason = "The justification behind a decision"
Alternative = "An option considered but not chosen"

[schema.relations]
chose = { head = "Person", tail = "Component" }
rejected = { head = "Person", tail = "Alternative" }
approved = { head = "Person", tail = "Decision" }

[tier2]
enabled = true

[tier2.dedup.aliases]
"Postgres" = ["PostgreSQL", "PG", "psql"]

[llm]
enabled = false   # opt-in only
```

## Project Structure

```
crates/
├── ctxgraph-core/       Core engine: types, storage, query, temporal
├── ctxgraph-extract/    Three-tier extraction pipeline (ONNX)
├── ctxgraph-embed/      Local embedding generation
├── ctxgraph-cli/        CLI binary
├── ctxgraph-mcp/        MCP server for AI agents
└── ctxgraph-sdk/        Re-export crate for embedding in Rust apps
```

## Roadmap

```
v0.1  Core Engine                   DONE    SQLite, FTS5, bi-temporal, CLI
v0.2  GLiNER2 Extraction            ▸       Unified NER + RE in one ONNX model
v0.3  MCP + Search                          Semantic embeddings, RRF, MCP server
v0.4  Tier 2 + Git Watch                   Dedup, coref, auto-capture commits
v0.5  Tier 3 + Bulk Ingest                 Ollama/API, JSONL/CSV import
v1.0  Production Ready                     Benchmarks, docs, pre-built binaries
```

### v0.1 — Core Engine (done)

- `ctxgraph-core` and `ctxgraph-cli` crates
- Episode, Entity, Edge types with builder pattern
- SQLite storage with embedded migrations and FTS5
- Bi-temporal model: `valid_from` / `valid_until` / `recorded_at`
- Edge invalidation (facts never deleted, only marked)
- Recursive CTE graph traversal
- CLI: `init`, `log`, `query`, `entities`, `decisions`, `stats`
- 24 tests passing, 0 clippy warnings

### v0.2 — GLiNER2 Unified Extraction (next)

The hardest engineering sprint. GLiNER2 (2025 EMNLP) handles entity + relation extraction in a single ONNX forward pass — one model (~200MB), one inference call, entities and relations out.

- `ctxgraph-extract` crate with ONNX Runtime integration
- Schema-driven extraction labels from `ctxgraph.toml`
- Temporal heuristic parser (ISO-8601, relative dates, fiscal quarters)
- Extraction benchmark: 50 annotated episodes, F1 ≥ 0.80
- Model download/cache/verify pipeline
- GLiREL available as optional precision mode

### v0.3 — MCP Server + Search (demo milestone)

The point where you can show it working in Claude Desktop / Cursor.

- `ctxgraph-embed` crate: all-MiniLM-L6-v2 for 384-dim embeddings
- `ctxgraph-mcp` crate: 5 tools over stdio JSON-RPC
- Reciprocal Rank Fusion: FTS5 + semantic + graph traversal
- Cold start UX: sparse graph hints, bootstrap suggestions

### v0.4 — Tier 2 + Git Watch (launch milestone)

Quality improvement + passive capture. Open source and launch after this.

- Coreference resolution (rule-based pronoun → entity)
- Fuzzy entity dedup (Jaro-Winkler + configurable aliases)
- `ctxgraph watch --git` auto-captures commit messages
- Extraction accuracy ~90% on semi-structured text

### v0.5 — Tier 3 + Bulk Ingest (user-guided)

Built based on feedback from real users after launch.

- Ollama / OpenAI-compatible LLM provider
- Contradiction detection, community summarization
- JSONL / CSV / stdin bulk import
- Built-in schemas: default, developer, support, finance
- JSON / CSV export

### v1.0 — Production Ready

- Criterion benchmarks at 1K / 10K / 100K episodes
- Rustdoc for all public APIs
- Pre-built binaries for Linux, macOS, Windows
- Published to crates.io

## Design Principles

1. **Zero infrastructure** — One binary, one SQLite file
2. **Offline-first** — No internet required after model download
3. **Privacy by default** — Nothing leaves your machine
4. **Progressive enhancement** — Each tier is additive and optional
5. **Schema-driven** — Extraction labels are user-defined
6. **Embeddable** — Rust library first, CLI second
7. **Append-only history** — Facts invalidated, never deleted
8. **Ship fast, iterate with users** — Demo in 5 weeks, launch in 6

## License

MIT
