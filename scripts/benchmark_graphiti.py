#!/usr/bin/env python3
"""Benchmark Graphiti extraction against the same 50 gold episodes used for ctxgraph.

Requires:
  - Neo4j running (docker: neo4j-graphiti)
  - OPENAI_API_KEY set
  - pip install graphiti-core neo4j

Usage:
  OPENAI_API_KEY=sk-... python scripts/benchmark_graphiti.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Load benchmark episodes ──────────────────────────────────────────────
EPISODES_PATH = Path(__file__).parent.parent / "crates/ctxgraph-extract/tests/fixtures/benchmark_episodes.json"


def load_episodes():
    with open(EPISODES_PATH) as f:
        return json.load(f)


# ── F1 scoring (same logic as ctxgraph benchmark) ───────────────────────
def entity_f1(expected, extracted):
    """Compute entity F1 based on name matching (case-insensitive)."""
    gold = {e["name"].lower() for e in expected}
    pred = {e.lower() for e in extracted}
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0
    recall = tp / len(gold) if gold else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def relation_f1(expected, extracted):
    """Compute relation F1 based on (head, relation, tail) matching (case-insensitive)."""
    def normalize_rel(r):
        return (r["head"].lower(), r["relation"].lower(), r["tail"].lower())

    gold = {normalize_rel(r) for r in expected}
    pred = {normalize_rel(r) for r in extracted}
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0
    recall = tp / len(gold) if gold else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Graphiti extraction ──────────────────────────────────────────────────
async def run_benchmark():
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "graphiti123")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    episodes = load_episodes()
    print(f"Loaded {len(episodes)} benchmark episodes")

    # Initialize Graphiti (uses OpenAI by default)
    client = Graphiti(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)

    try:
        await client.build_indices_and_constraints()

        entity_f1s = []
        relation_f1s = []
        total_cost_tokens = 0

        for i, ep in enumerate(episodes):
            text = ep["text"]
            expected_ents = ep["expected_entities"]
            expected_rels = ep["expected_relations"]

            print(f"\n── Episode {i+1}/{len(episodes)} ──")
            print(f"  Text: {text[:80]}...")

            try:
                result = await client.add_episode(
                    name=f"benchmark_ep{i+1}",
                    episode_body=text,
                    source_description="ctxgraph benchmark comparison",
                    reference_time=datetime.now(timezone.utc),
                    source=EpisodeType.text,
                    group_id=f"benchmark_{i+1}",
                )

                # Extract entity names from result
                extracted_entity_names = [node.name for node in result.nodes]

                # Build UUID→name map for edge resolution
                uuid_to_name = {str(node.uuid): node.name for node in result.nodes}

                # Extract relations from result edges
                extracted_relations = []
                for edge in result.edges:
                    src = uuid_to_name.get(str(edge.source_node_uuid), str(edge.source_node_uuid))
                    tgt = uuid_to_name.get(str(edge.target_node_uuid), str(edge.target_node_uuid))
                    rel_name = edge.name if edge.name else "unknown"
                    extracted_relations.append({
                        "head": src,
                        "relation": rel_name,
                        "tail": tgt,
                    })

                # Compute F1 scores
                ef1 = entity_f1(expected_ents, extracted_entity_names)
                rf1 = relation_f1(expected_rels, extracted_relations)

                entity_f1s.append(ef1)
                relation_f1s.append(rf1)

                print(f"  Entities extracted: {extracted_entity_names}")
                print(f"  Relations extracted: {[(r['head'], r['relation'], r['tail']) for r in extracted_relations]}")
                print(f"  Entity F1: {ef1:.3f}  |  Relation F1: {rf1:.3f}")

                # Track token usage if available
                if hasattr(client, 'token_tracker'):
                    tracker = client.token_tracker
                    if hasattr(tracker, 'total_tokens'):
                        total_cost_tokens = tracker.total_tokens

            except Exception as e:
                print(f"  ERROR: {e}")
                entity_f1s.append(0.0)
                relation_f1s.append(0.0)

        # ── Summary ──────────────────────────────────────────────────────
        avg_ef1 = sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0
        avg_rf1 = sum(relation_f1s) / len(relation_f1s) if relation_f1s else 0
        combined = (avg_ef1 + avg_rf1) / 2

        print("\n" + "=" * 60)
        print("GRAPHITI BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Episodes:          {len(episodes)}")
        print(f"Avg Entity F1:     {avg_ef1:.4f}")
        print(f"Avg Relation F1:   {avg_rf1:.4f}")
        print(f"Combined F1:       {combined:.4f}")
        print(f"Token usage:       ~{total_cost_tokens}")
        print()
        print("Per-episode scores:")
        for i, (ef, rf) in enumerate(zip(entity_f1s, relation_f1s)):
            marker = " <<<" if (ef + rf) / 2 < 0.5 else ""
            print(f"  ep{i+1:02d}: entity={ef:.3f} rel={rf:.3f} combined={(ef+rf)/2:.3f}{marker}")

        # Save results to JSON for comparison
        results = {
            "system": "graphiti",
            "version": "0.28.2",
            "llm": "gpt-4o (default)",
            "avg_entity_f1": avg_ef1,
            "avg_relation_f1": avg_rf1,
            "combined_f1": combined,
            "per_episode": [
                {"episode": i+1, "entity_f1": ef, "relation_f1": rf}
                for i, (ef, rf) in enumerate(zip(entity_f1s, relation_f1s))
            ],
        }
        out_path = Path(__file__).parent / "graphiti_benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
