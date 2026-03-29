# ctxgraph

**Privacy-first knowledge graph engine for AI agents.**

Extracts entities and relations from any text. Builds a temporal knowledge graph. Works locally with zero API keys — and when it does call an LLM, it makes one call per episode instead of Graphiti's six.

```bash
brew install rohansx/tap/ctxgraph
ctxgraph init && ctxgraph models download
ctxgraph log "Migrated auth from Redis sessions to JWT. Chose JWT for stateless scaling."
ctxgraph query "Why did we move away from Redis?"
```

---

## Why ctxgraph?

Every knowledge graph engine requires an LLM for every operation. Graphiti makes 6 API calls per episode. Mem0 calls GPT-4 on every add/search. Microsoft GraphRAG is so expensive they put a cost warning in their README.

ctxgraph runs a tiered extraction pipeline: local ONNX models handle most episodes at zero cost, and only escalates to an LLM when local confidence is low. One call, not six. PII stripped before it leaves your machine.

**Validated on 20 unseen episodes across 12 domains:**

| | ctxgraph | Graphiti | Mem0 | LightRAG |
|---|---|---|---|---|
| **Combined F1** | **0.678** | 0.287 | N/A (no KG eval) | N/A (RAG, no KG) |
| **Entity F1** | **0.854** | 0.468 | — | — |
| **Relation F1** | **0.502** | 0.106 | — | — |
| LLM calls per episode | **1** | 6 | 1-2 | 1+ |
| Works without LLM? | **Yes** | No | No | No |
| Works offline? | **Yes** | No | No | Partial |
| Query latency | **<15ms** | ~300ms | ~100ms | ~200ms |
| Infrastructure | **SQLite** | Neo4j+Docker | Vector DB | Varies |
| Language | **Rust** | Python | Python | Python |
| Privacy (PII protection) | **CloakPipe (v0.8)** | None | None | None |

---

## How It Works

```
Text comes in
    |
    v
[Tier 1] GLiNER entities + GLiREL relations (local ONNX, FREE, ~10ms)
    |
    v
Confidence gate: good enough?
    |
    +-- YES (tech text, ~70%) --> Graph. Done. $0.
    |
    +-- NO (cross-domain)    --> LLM (1 call) --> Graph.  [CloakPipe PII stripping in v0.8]
```

| Tier | What | Cost | Latency |
|---|---|---|---|
| **Local ONNX** | GLiNER (entities) + GLiREL (relations) | $0 | ~10ms |
| **LLM fallback** | One call for entities + relations when local isn't confident | ~$0.0003/ep | ~3-5s |
| **Dedup** | Jaro-Winkler similarity + alias table (local) | $0 | <1ms |
| **Search** | FTS5 + semantic + graph walk, fused via RRF (local) | $0 | <15ms |

Graphiti does ALL of these via LLM: entity extraction, deduplication, relation extraction, contradiction detection, summarization, community detection. Six calls. Every episode.

---

## Competitive Landscape

### Knowledge Graph Engines

| | ctxgraph | Graphiti | Cognee | WhyHow.AI |
|---|---|---|---|---|
| **Extraction** | Local ONNX + LLM fallback | LLM only (6 calls/ep) | LLM only | LLM only (OpenAI) |
| **Graph DB** | SQLite (embedded) | Neo4j/FalkorDB | Neo4j/Kuzu | MongoDB Atlas |
| **Works offline?** | **Yes** | No | No | No |
| **Temporal queries** | **Bi-temporal** | Bi-temporal | No | No |
| **MCP support** | Yes | Yes | Yes | No |
| **Language** | Rust (single binary) | Python | Python | Python |
| **Schema-driven** | Yes (ctxgraph.toml) | Yes (prescribed ontology) | Yes | Yes |
| **Cost/1000 eps** | **$0.30** | $1.80 | ~$1.50 | ~$2+ |
| **Stars** | Early | 24K | 15K | 900 |

### Agent Memory Systems

| | ctxgraph | Mem0 | Basic Memory | mcp-memory-service |
|---|---|---|---|---|
| **Entity extraction** | **Automated (ONNX+LLM)** | LLM-only | None (manual) | None (manual) |
| **Relation extraction** | **Automated (GLiREL+LLM)** | Limited | None | Manual typed edges |
| **Knowledge graph** | **Yes (temporal)** | Optional (Neptune) | Semantic links | Basic typed edges |
| **Works without LLM?** | **Yes** | No | Yes | Yes |
| **Query latency** | **<15ms** | ~100ms | ~10ms | ~5ms |
| **Dedup** | Jaro-Winkler + aliases | LLM-based | None | None |
| **Cost** | **$0 (local) / $0.30 (hybrid)** | LLM cost per op | $0 | $0 |
| **Stars** | Early | 51K | 2.7K | 1.6K |

### Graph-Enhanced RAG

| | ctxgraph | LightRAG | Microsoft GraphRAG | nano-graphrag |
|---|---|---|---|---|
| **Purpose** | Knowledge graph engine | RAG retrieval | Document summarization | Lightweight GraphRAG |
| **Incremental updates** | **Yes** | Yes | **No** (batch only) | No |
| **Temporal awareness** | **Yes (bi-temporal)** | No | No | No |
| **LLM per query** | **No** | Yes | Yes | Yes |
| **Offline capable** | **Yes** | Partial (Ollama) | No | Partial |
| **Cost/1000 docs** | **$0.30** | ~$1-5 | ~$10-50 | ~$1-5 |
| **Stars** | Early | 31K | 32K | 3.8K |

### What Makes ctxgraph Unique

**No other tool has all of these:**
1. **Automated extraction that works offline** — GLiNER + GLiREL via ONNX. Mem0 and Basic Memory have no extraction. Graphiti/Cognee/LightRAG always need an LLM.
2. **Typed relations, not free-form** — Graphiti produces `SWITCHED_TO`, `CAUSED_OOM_KILLS`. ctxgraph produces `replaced`, `caused` — queryable by type.
3. **Bi-temporal history** — Only ctxgraph and Graphiti have this. Every other tool is current-state only.
4. **PII protection (v0.8)** — CloakPipe will strip PII before LLM calls. No competitor has this planned.
5. **Single Rust binary** — Every competitor is Python with pip/Docker/Neo4j. ctxgraph is `cargo install`.
6. **6x fewer LLM calls** — 1 call vs Graphiti's 6. Same model, better results.

---

## Validated Benchmark

Tested on 20 completely new episodes across 12 domains. Both systems use GPT-4o-mini. Neither system has seen this data before.

### Overall Results

| | Graphiti | ctxgraph |
|---|---|---|
| **Entity F1** | 0.468 | **0.854** |
| **Relation F1** | 0.106 | **0.502** |
| **Combined F1** | 0.287 | **0.678** |
| Time/episode | 18.2s | **5.1s** |
| LLM calls (20 eps) | ~120 | **20** |

### Per-Domain (ctxgraph scores)

| Domain | Entity F1 | Relation F1 | Combined |
|---|---|---|---|
| Tech (Slack, PRs, ADRs, incidents) | 0.877 | 0.530 | 0.703 |
| Finance | 0.769 | 0.600 | 0.685 |
| Healthcare | 0.909 | 0.286 | 0.598 |
| Legal | 0.857 | 0.500 | 0.679 |
| Manufacturing | 0.667 | 0.333 | 0.500 |
| Education | 0.667 | 0.286 | 0.477 |
| Real Estate | 1.000 | 0.667 | 0.834 |
| E-commerce | 0.800 | 0.500 | 0.650 |
| Logistics | 0.889 | 0.667 | 0.778 |
| Gaming | 1.000 | 1.000 | 1.000 |
| Government | 0.909 | 0.000 | 0.455 |

### LLM Model Comparison (cross-domain)

| LLM | Hostable locally? | Cross-domain F1 | Cost/1000 eps |
|---|---|---|---|
| None (local only) | N/A | 0.325 | $0 |
| Llama 3.2 3B (Ollama) | Yes (8GB) | 0.472 | $0 |
| Qwen 2.5 7B (Ollama) | Yes (16GB) | 0.508 | $0 |
| GPT-4o-mini | Cloud | **0.650** | $0.30 |
| Claude 3.5 Haiku | Cloud | 0.611 | $0.09 |
| Gemini 2.0 Flash | Cloud | 0.552 | $0.05 |

### Query Performance

| | ctxgraph | Graphiti |
|---|---|---|
| Full-text search | **<1ms** | ~50ms |
| Semantic search | **3-5ms** | ~100ms |
| Graph traversal (2-3 hops) | **<5ms** | 5-50ms |
| Fused search (RRF) | **<15ms** | ~300ms |

---

## Key Features

- **Tiered extraction** — Local ONNX first, LLM only when needed
- **Privacy (v0.8)** — CloakPipe PII stripping before cloud calls coming soon
- **Zero infrastructure** — One binary, one SQLite file
- **Any LLM** — Ollama, NVIDIA NIM (free), OpenRouter, OpenAI, Anthropic
- **Bi-temporal** — Time-travel queries, fact invalidation
- **Schema-driven** — Entity/relation types via `ctxgraph.toml`
- **MCP server** — Claude Code, Cursor, Cline, any MCP client
- **Embeddable** — Rust library, CLI, or MCP server
- **Entity dedup** — Jaro-Winkler + alias table across episodes

---

## Installation

```bash
# Homebrew
brew install rohansx/tap/ctxgraph

# Or build from source (Rust 1.85+)
cargo install ctxgraph-cli
```

## Quick Start

```bash
ctxgraph models download              # one-time ONNX model download
ctxgraph init                         # initialize graph in current dir
ctxgraph log "Alice chose PostgreSQL"  # extract + store
ctxgraph query "why PostgreSQL?"       # search the graph
```

### Optional: LLM for cross-domain quality

```toml
# ctxgraph.toml
[llm]
provider = "openrouter"
[llm.openrouter]
model = "openai/gpt-4o-mini"
api_key_env = "OPENROUTER_API_KEY"

[privacy]
# cloakpipe = true                    # coming in v0.8
```

Without this, everything runs locally at 0.846 F1 on tech text.

---

## MCP Server

```json
{
  "mcpServers": {
    "ctxgraph": { "command": "ctxgraph-mcp" }
  }
}
```

| Tool | Description |
|---|---|
| `ctxgraph_add_episode` | Record a decision or event |
| `ctxgraph_search` | Fused FTS5 + semantic + graph search |
| `ctxgraph_traverse` | Walk the graph from an entity |
| `ctxgraph_find_precedents` | Find similar past events |
| `ctxgraph_list_entities` | List entities with filters |
| `ctxgraph_export_graph` | Export entities and edges |

## Rust SDK

```rust
let graph = ctxgraph::Graph::init(".ctxgraph")?;
graph.add_episode(Episode::builder("Chose Postgres for billing").build())?;
let results = graph.search("why Postgres?", 10)?;
```

## Project Structure

```
crates/
+-- ctxgraph-core/       Types, storage, query, temporal
+-- ctxgraph-extract/    Tiered extraction (ONNX + LLM)
+-- ctxgraph-embed/      Local embeddings
+-- ctxgraph-cli/        CLI binary
+-- ctxgraph-mcp/        MCP server
+-- ctxgraph-sdk/        Rust SDK
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
