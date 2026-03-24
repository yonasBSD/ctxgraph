#!/usr/bin/env python3
"""Compare ctxgraph and Graphiti benchmark results with semantic relation mapping.

Maps Graphiti's free-form relation names to ctxgraph's schema taxonomy using
keyword matching, then re-scores both entity and relation F1.

Usage:
  python scripts/compare_benchmarks.py
"""

import json
import re
from pathlib import Path

EPISODES_PATH = Path(__file__).parent.parent / "crates/ctxgraph-extract/tests/fixtures/benchmark_episodes.json"
GRAPHITI_LOG = Path("/tmp/graphiti_benchmark_full.log")

# ── Semantic mapping: Graphiti free-form → ctxgraph taxonomy ─────────────
RELATION_MAP = {
    # chose / selected / adopted
    r"CHOSE|CHOOSE|SELECTED|DECIDED_TO_USE|DECIDED_TO_ADOPT|PROPOSED_ADOPTING|CHOSE_FOR|CHOOSES_AS": "chose",
    # rejected
    r"REJECTED|DOES_NOT_CHOOSE|DID_NOT_CHOOSE|REJECTS": "rejected",
    # replaced
    r"REPLACED|MIGRATED_TO|MIGRATED_FROM|REWRITTEN_TO|REWRITTEN_FROM|WAS_REWRITTEN|CONVERTED_TO|UPGRADED_TO|UPGRADED_FROM|SWITCHED_FROM|SWITCHED_TO|WAS_CONVERTED": "replaced",
    # depends_on
    r"DEPENDS_ON|USES|BACKED_BY|CONNECTS_TO|INTEGRATED_IN|INTEGRATED_WITH|IS_MANAGED_BY|USES_LIBRARY|RUNS_ON|ENABLED_ON|PERSISTED_TO|PROVISION_ON|LEVERAGES|ROUTES_TRAFFIC_TO": "depends_on",
    # fixed
    r"FIXED|IDENTIFIED_MISSING|PATCHED|RESOLVED|RESOLVED_BY": "fixed",
    # introduced
    r"INTRODUCED|ADDED_TO|IMPLEMENTED|ENABLED_AT|IMPLEMENTS_PATTERN|USES_PATTERN|PROPOSED": "introduced",
    # deprecated
    r"DEPRECATED|FORBIDDEN|STOPPED_USING|IS_FORBIDDEN": "deprecated",
    # caused
    r"CAUSED|IMPROVED_BY|CRASHES_WHEN|SPIKED_AFTER|AFFECTED|RESULTS_IN|REDUCED|IMPROVED|LED_TO": "caused",
    # constrained_by
    r"CONSTRAINED_BY|MUST_COMPLY|MANDATED|REQUIRES|ENFORCES|HAS_CONSTRAINT|SCOPED_PER|CAPPED_AT": "constrained_by",
}


def map_relation(graphiti_rel: str) -> str | None:
    """Map a Graphiti free-form relation name to ctxgraph taxonomy."""
    upper = graphiti_rel.upper().replace(" ", "_")
    for pattern, ctxgraph_type in RELATION_MAP.items():
        if re.search(pattern, upper):
            return ctxgraph_type
    return None


# ── Parsing ──────────────────────────────────────────────────────────────
def parse_graphiti_log(log_path: Path) -> list[dict]:
    """Parse the Graphiti benchmark log into per-episode results."""
    text = log_path.read_text()
    episodes = []

    # Split by episode markers
    parts = re.split(r"── Episode (\d+)/50 ──", text)

    for i in range(1, len(parts), 2):
        ep_num = int(parts[i])
        body = parts[i + 1] if i + 1 < len(parts) else ""

        # Extract entities
        ent_match = re.search(r"Entities extracted: \[(.*?)\]", body)
        entities = []
        if ent_match:
            entities = [e.strip().strip("'\"") for e in ent_match.group(1).split("', '")]

        # Extract relations
        rel_match = re.search(r"Relations extracted: \[(.*?)\]", body)
        relations = []
        if rel_match:
            for m in re.finditer(r"\('([^']+)', '([^']+)', '([^']+)'\)", rel_match.group(1)):
                relations.append({
                    "head": m.group(1),
                    "relation": m.group(2),
                    "tail": m.group(3),
                })

        episodes.append({
            "episode": ep_num,
            "entities": entities,
            "relations": relations,
        })

    return episodes


def entity_f1(expected, extracted_names):
    gold = {e["name"].lower() for e in expected}
    pred = {n.lower() for n in extracted_names}
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0
    r = tp / len(gold) if gold else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def relation_f1(expected, extracted):
    def norm(r):
        return (r["head"].lower(), r["relation"].lower(), r["tail"].lower())
    gold = {norm(r) for r in expected}
    pred = {norm(r) for r in extracted}
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0
    r = tp / len(gold) if gold else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    episodes = json.loads(EPISODES_PATH.read_text())
    graphiti_results = parse_graphiti_log(GRAPHITI_LOG)

    print("=" * 80)
    print("BENCHMARK COMPARISON: ctxgraph v0.6.0 vs Graphiti v0.28.2 (gpt-4o)")
    print("=" * 80)
    print()

    # ── Raw comparison (exact match) ─────────────────────────────────────
    graphiti_ef1s = []
    graphiti_rf1s_raw = []
    graphiti_rf1s_mapped = []

    for ep_gold, ep_graphiti in zip(episodes, graphiti_results):
        ef1 = entity_f1(ep_gold["expected_entities"], ep_graphiti["entities"])
        graphiti_ef1s.append(ef1)

        # Raw relation F1 (exact match — will be ~0)
        rf1_raw = relation_f1(ep_gold["expected_relations"], ep_graphiti["relations"])
        graphiti_rf1s_raw.append(rf1_raw)

        # Mapped relation F1 (semantic mapping)
        mapped_rels = []
        for rel in ep_graphiti["relations"]:
            mapped_type = map_relation(rel["relation"])
            if mapped_type:
                mapped_rels.append({
                    "head": rel["head"],
                    "relation": mapped_type,
                    "tail": rel["tail"],
                })
        rf1_mapped = relation_f1(ep_gold["expected_relations"], mapped_rels)
        graphiti_rf1s_mapped.append(rf1_mapped)

    avg_ef1 = sum(graphiti_ef1s) / len(graphiti_ef1s)
    avg_rf1_raw = sum(graphiti_rf1s_raw) / len(graphiti_rf1s_raw)
    avg_rf1_mapped = sum(graphiti_rf1s_mapped) / len(graphiti_rf1s_mapped)

    print("GRAPHITI RESULTS:")
    print(f"  Avg Entity F1:           {avg_ef1:.4f}")
    print(f"  Avg Relation F1 (raw):   {avg_rf1_raw:.4f}")
    print(f"  Avg Relation F1 (mapped):{avg_rf1_mapped:.4f}")
    print(f"  Combined F1 (raw):       {(avg_ef1 + avg_rf1_raw) / 2:.4f}")
    print(f"  Combined F1 (mapped):    {(avg_ef1 + avg_rf1_mapped) / 2:.4f}")
    print()

    # ── ctxgraph results (from benchmark test) ───────────────────────────
    print("CTXGRAPH RESULTS (from cargo test --ignored):")
    print(f"  Avg Entity F1:           0.8372")
    print(f"  Avg Relation F1:         0.7628")
    print(f"  Combined F1:             0.8000")
    print()

    # ── Comparison table ─────────────────────────────────────────────────
    print("=" * 80)
    print(f"{'Metric':<30} {'ctxgraph':>12} {'Graphiti':>12} {'Graphiti*':>12}")
    print(f"{'':30} {'(local)':>12} {'(raw)':>12} {'(mapped)':>12}")
    print("-" * 80)
    print(f"{'Avg Entity F1':<30} {'0.8372':>12} {avg_ef1:>12.4f} {avg_ef1:>12.4f}")
    print(f"{'Avg Relation F1':<30} {'0.7628':>12} {avg_rf1_raw:>12.4f} {avg_rf1_mapped:>12.4f}")
    print(f"{'Combined F1':<30} {'0.8000':>12} {(avg_ef1 + avg_rf1_raw)/2:>12.4f} {(avg_ef1 + avg_rf1_mapped)/2:>12.4f}")
    print(f"{'API calls':<30} {'0':>12} {'~200+':>12} {'~200+':>12}")
    print(f"{'Cost per run':<30} {'$0':>12} {'~$2-5':>12} {'~$2-5':>12}")
    print(f"{'Latency (50 eps)':<30} {'~2s':>12} {'~8min':>12} {'~8min':>12}")
    print(f"{'Requires internet':<30} {'No':>12} {'Yes':>12} {'Yes':>12}")
    print("=" * 80)
    print()
    print("* 'mapped' = Graphiti's free-form relations mapped to ctxgraph's taxonomy")
    print("  using keyword heuristics (generous to Graphiti)")
    print()

    # ── Per-episode details ──────────────────────────────────────────────
    print("Per-episode mapped relation F1 (Graphiti with semantic mapping):")
    mapped_nonzero = 0
    for i, (rf1m, ef1) in enumerate(zip(graphiti_rf1s_mapped, graphiti_ef1s)):
        if rf1m > 0:
            mapped_nonzero += 1
        marker = " <<<" if rf1m == 0.0 else ""
        print(f"  ep{i+1:02d}: entity={ef1:.3f} rel_mapped={rf1m:.3f}{marker}")

    print(f"\nEpisodes with >0 mapped relation F1: {mapped_nonzero}/{len(graphiti_rf1s_mapped)}")

    # ── Save comparison JSON ─────────────────────────────────────────────
    comparison = {
        "ctxgraph": {
            "version": "0.6.0",
            "infrastructure": "fully local (ONNX models)",
            "avg_entity_f1": 0.8372,
            "avg_relation_f1": 0.7628,
            "combined_f1": 0.8000,
            "api_calls": 0,
            "cost": "$0",
            "latency_50_episodes": "~2s",
        },
        "graphiti": {
            "version": "0.28.2",
            "infrastructure": "Neo4j + OpenAI gpt-4o",
            "avg_entity_f1": round(avg_ef1, 4),
            "avg_relation_f1_raw": round(avg_rf1_raw, 4),
            "avg_relation_f1_mapped": round(avg_rf1_mapped, 4),
            "combined_f1_raw": round((avg_ef1 + avg_rf1_raw) / 2, 4),
            "combined_f1_mapped": round((avg_ef1 + avg_rf1_mapped) / 2, 4),
            "api_calls": "~200+ (OpenAI)",
            "cost": "~$2-5",
            "latency_50_episodes": "~8min",
        },
        "per_episode": [
            {
                "episode": i + 1,
                "graphiti_entity_f1": round(ef1, 4),
                "graphiti_rel_f1_raw": round(rf1r, 4),
                "graphiti_rel_f1_mapped": round(rf1m, 4),
            }
            for i, (ef1, rf1r, rf1m) in enumerate(
                zip(graphiti_ef1s, graphiti_rf1s_raw, graphiti_rf1s_mapped)
            )
        ],
    }

    out_path = Path(__file__).parent / "benchmark_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison saved to {out_path}")


if __name__ == "__main__":
    main()
