#!/usr/bin/env python3
"""
Head-to-head: Graphiti vs ctxgraph on identical real-world tech episodes.

Usage:
    # With NVIDIA router (free):
    python scripts/graphiti_vs_ctxgraph.py

    # With OpenAI:
    OPENAI_API_KEY=sk-xxx python scripts/graphiti_vs_ctxgraph.py --model gpt-4o-mini
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

EPISODES_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "crates/ctxgraph-extract/tests/fixtures/real_world_tech_episodes.json",
)

with open(EPISODES_PATH) as f:
    EPISODES = json.load(f)


def fuzzy_match(a: str, b: str) -> bool:
    al, bl = a.lower().strip(), b.lower().strip()
    return al == bl or al in bl or bl in al


def compute_f1_fuzzy(predicted, expected):
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    matched = [False] * len(expected)
    tp = 0.0
    for pred in predicted:
        for i, exp in enumerate(expected):
            if not matched[i] and fuzzy_match(pred, exp):
                tp += 1.0
                matched[i] = True
                break
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(expected) if expected else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


async def run_graphiti(model_name: str, base_url: str, api_key: str):
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import OpenAIClient, LLMConfig

    print(f"\n{'='*60}")
    print(f"GRAPHITI — {model_name}")
    print(f"{'='*60}")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model_name}")

    config_kwargs = {
        "api_key": api_key,
        "model": model_name,
        "small_model": model_name,
    }
    if base_url:
        config_kwargs["base_url"] = base_url
    llm_config = LLMConfig(**config_kwargs)
    llm_client = OpenAIClient(config=llm_config)

    graphiti = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "password123",
        llm_client=llm_client,
    )

    try:
        await graphiti.build_indices_and_constraints()
    except Exception:
        pass  # indices already exist

    total_ent_f1 = 0.0
    total_rel_f1 = 0.0
    total_time = 0.0
    results = []

    for i, ep in enumerate(EPISODES):
        start = time.time()
        try:
            await graphiti.add_episode(
                name=f"realworld_ep_{i}_{ep['source']}",
                episode_body=ep["text"],
                source_description=ep["source"],
                reference_time=datetime.now(),
            )
            elapsed = time.time() - start
            total_time += elapsed

            # Search for what Graphiti extracted
            search_results = await graphiti.search(ep["text"][:80], num_results=20)

            pred_entities = []
            pred_facts = []
            for r in search_results:
                if hasattr(r, "name") and r.name:
                    pred_entities.append(r.name.lower())
                if hasattr(r, "fact") and r.fact:
                    pred_facts.append(r.fact.lower())

            exp_entities = [e["name"].lower() for e in ep["expected_entities"]]

            _, _, ent_f1 = compute_f1_fuzzy(pred_entities, exp_entities)

            # Relation matching: check if Graphiti's free-form facts mention expected head+tail
            exp_relations = ep["expected_relations"]
            rel_matches = 0
            for er in exp_relations:
                head, tail = er["head"].lower(), er["tail"].lower()
                for fact in pred_facts:
                    if head in fact and tail in fact:
                        rel_matches += 1
                        break
            rel_recall = rel_matches / len(exp_relations) if exp_relations else 0.0

            total_ent_f1 += ent_f1
            total_rel_f1 += rel_recall

            print(
                f"  [{ep['source']:>15}] ep{i}: entity={ent_f1:.3f} rel={rel_recall:.3f} "
                f"time={elapsed:.1f}s found={len(pred_entities)} ents, {len(pred_facts)} facts"
            )
            results.append({"source": ep["source"], "ent_f1": ent_f1, "rel_f1": rel_recall, "time": elapsed})

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            err_msg = str(e)[:120]
            print(f"  [{ep['source']:>15}] ep{i}: ERROR ({elapsed:.1f}s): {err_msg}")
            results.append({"source": ep["source"], "ent_f1": 0, "rel_f1": 0, "time": elapsed, "error": err_msg})

    n = len(EPISODES)
    avg_ent = total_ent_f1 / n
    avg_rel = total_rel_f1 / n
    combined = (avg_ent + avg_rel) / 2

    print(f"\n  --- GRAPHITI TOTALS ---")
    print(f"  Avg entity F1:       {avg_ent:.3f}")
    print(f"  Avg relation recall: {avg_rel:.3f}")
    print(f"  Combined:            {combined:.3f}")
    print(f"  Total time:          {total_time:.1f}s ({total_time/n:.1f}s/episode)")
    print(f"  LLM calls:           ~{n * 6} (6 per episode)")

    await graphiti.close()
    return {"avg_ent": avg_ent, "avg_rel": avg_rel, "combined": combined, "total_time": total_time}


async def main():
    # Determine LLM backend
    model = os.environ.get("GRAPHITI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL", "")  # empty = OpenAI default
    api_key = os.environ.get("OPENAI_API_KEY", "")

    # Allow CLI override
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]

    print("=" * 60)
    print("GRAPHITI vs CTXGRAPH — Real-World Tech Data")
    print("=" * 60)
    print(f"Episodes: {len(EPISODES)}")

    graphiti_results = await run_graphiti(model, base_url, api_key)

    print(f"\n{'='*60}")
    print("CTXGRAPH (from benchmark runs)")
    print(f"{'='*60}")
    print(f"  Local only:        combined=0.350 (entity=0.545, relation=0.154)")
    print(f"  + NVIDIA reasoning: combined=0.342 (entity=0.542, relation=0.142)")
    print(f"  + Gemini Flash:    combined=0.330 (entity=0.553, relation=0.107)")
    print(f"  + Claude Haiku:    combined=0.375 (entity=0.631, relation=0.118)")
    print(f"  LLM calls:         0-8 (only when gate fires)")

    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'='*60}")
    g = graphiti_results
    print(f"  {'':>20} {'Graphiti':>12} {'ctxgraph':>12} {'ctxgraph':>12}")
    print(f"  {'':>20} {'('+model+')':>12} {'(local)':>12} {'(+Haiku)':>12}")
    print(f"  {'Entity F1':>20} {g['avg_ent']:>12.3f} {'0.545':>12} {'0.631':>12}")
    print(f"  {'Relation F1':>20} {g['avg_rel']:>12.3f} {'0.154':>12} {'0.118':>12}")
    print(f"  {'Combined':>20} {g['combined']:>12.3f} {'0.350':>12} {'0.375':>12}")
    print(f"  {'Time/episode':>20} {g['total_time']/len(EPISODES):>11.1f}s {'0.01':>11}s {'0.5':>11}s")
    print(f"  {'LLM calls':>20} {'~48':>12} {'0':>12} {'~8':>12}")
    print(f"  {'Cost (1000 eps)':>20} {'~$30':>12} {'$0':>12} {'~$0.09':>12}")


if __name__ == "__main__":
    asyncio.run(main())
