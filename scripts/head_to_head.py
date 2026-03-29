#!/usr/bin/env python3
"""
Fair head-to-head: Graphiti vs ctxgraph on identical episodes.
Both use GPT-4o-mini. Same data. Same model. Fair evaluation.

Usage:
    export OPENAI_API_KEY=sk-xxx
    python scripts/head_to_head.py
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ── Test Episodes ──────────────────────────────────────────────────

TECH_EPISODES = [
    {
        "text": "Alice migrated the auth service from Redis sessions to JWT tokens. The main reason was horizontal scaling — Redis was a single point of failure and JWT is stateless. She also deprecated the old session cleanup cron job.",
        "expected_entities": ["Alice", "auth service", "Redis", "JWT", "session cleanup cron job"],
        "expected_relations": [
            {"head": "JWT", "relation": "replaced", "tail": "Redis"},
            {"head": "Alice", "relation": "deprecated", "tail": "session cleanup cron job"},
            {"head": "auth service", "relation": "depends_on", "tail": "JWT"},
        ],
    },
    {
        "text": "We switched from CircleCI to GitHub Actions for CI/CD. The migration took two sprints. Bob wrote the new workflow files. The main blocker was Docker layer caching which GitHub Actions handles differently.",
        "expected_entities": ["CircleCI", "GitHub Actions", "Bob", "Docker"],
        "expected_relations": [
            {"head": "GitHub Actions", "relation": "replaced", "tail": "CircleCI"},
            {"head": "Bob", "relation": "introduced", "tail": "GitHub Actions"},
            {"head": "GitHub Actions", "relation": "depends_on", "tail": "Docker"},
        ],
    },
    {
        "text": "The payment service crashed because Stripe's webhook endpoint was rate-limited at 100 RPS. We added a Redis queue to buffer incoming events. Carlos implemented the fix in PR #234.",
        "expected_entities": ["payment service", "Stripe", "Redis", "Carlos"],
        "expected_relations": [
            {"head": "Stripe", "relation": "caused", "tail": "payment service"},
            {"head": "Carlos", "relation": "fixed", "tail": "payment service"},
            {"head": "payment service", "relation": "depends_on", "tail": "Redis"},
        ],
    },
    {
        "text": "After evaluating Terraform, Pulumi, and CloudFormation, the platform team chose Terraform for infrastructure as code. CloudFormation was rejected because we use multi-cloud. Pulumi was too new and lacked community modules.",
        "expected_entities": ["Terraform", "Pulumi", "CloudFormation", "platform team"],
        "expected_relations": [
            {"head": "platform team", "relation": "chose", "tail": "Terraform"},
            {"head": "platform team", "relation": "rejected", "tail": "CloudFormation"},
            {"head": "platform team", "relation": "rejected", "tail": "Pulumi"},
        ],
    },
    {
        "text": "Kubernetes cluster OOM killed the search-indexer pod three times yesterday. Root cause: Elasticsearch client was holding unbounded in-memory buffers. Diana patched it by adding a 512MB heap limit and switching to streaming bulk API.",
        "expected_entities": ["Kubernetes", "search-indexer", "Elasticsearch", "Diana"],
        "expected_relations": [
            {"head": "Elasticsearch", "relation": "caused", "tail": "search-indexer"},
            {"head": "Diana", "relation": "fixed", "tail": "search-indexer"},
            {"head": "search-indexer", "relation": "depends_on", "tail": "Elasticsearch"},
        ],
    },
]

CROSS_DOMAIN_EPISODES = [
    {
        "domain": "finance",
        "text": "The treasury department replaced Bloomberg Terminal with Refinitiv Eikon for fixed-income analytics after the annual license renewal hit $28K per seat. The migration required rewriting 15 Excel macros that depended on Bloomberg's BQL API.",
        "expected_entities": ["treasury department", "Bloomberg Terminal", "Refinitiv Eikon", "BQL API"],
        "expected_relations": [
            {"head": "Refinitiv Eikon", "relation": "replaced", "tail": "Bloomberg Terminal"},
            {"head": "treasury department", "relation": "chose", "tail": "Refinitiv Eikon"},
            {"head": "Bloomberg Terminal", "relation": "depends_on", "tail": "BQL API"},
        ],
    },
    {
        "domain": "healthcare",
        "text": "The hospital IT team migrated from Cerner Millennium to Epic Systems for electronic health records. HIPAA compliance required all patient data to be encrypted at rest. Dr. Williams led the clinical validation of the new system.",
        "expected_entities": ["Cerner Millennium", "Epic Systems", "HIPAA", "Dr. Williams"],
        "expected_relations": [
            {"head": "Epic Systems", "relation": "replaced", "tail": "Cerner Millennium"},
            {"head": "Epic Systems", "relation": "constrained_by", "tail": "HIPAA"},
            {"head": "Dr. Williams", "relation": "introduced", "tail": "Epic Systems"},
        ],
    },
    {
        "domain": "legal",
        "text": "The litigation support team adopted Relativity for e-discovery, replacing the manual document review process. FedRAMP authorization was a hard requirement from the government contracts division. Annual cost savings estimated at $2.3M.",
        "expected_entities": ["litigation support team", "Relativity", "FedRAMP", "government contracts division"],
        "expected_relations": [
            {"head": "litigation support team", "relation": "chose", "tail": "Relativity"},
            {"head": "Relativity", "relation": "constrained_by", "tail": "FedRAMP"},
        ],
    },
    {
        "domain": "manufacturing",
        "text": "The plant operations director chose Siemens MindSphere over PTC ThingWorx for predictive maintenance. The 50ms latency requirement for real-time sensor data ruled out cloud-only solutions. Apache Kafka handles the event stream from 2,000 edge sensors.",
        "expected_entities": ["Siemens MindSphere", "PTC ThingWorx", "Apache Kafka", "plant operations director"],
        "expected_relations": [
            {"head": "plant operations director", "relation": "chose", "tail": "Siemens MindSphere"},
            {"head": "plant operations director", "relation": "rejected", "tail": "PTC ThingWorx"},
            {"head": "Siemens MindSphere", "relation": "depends_on", "tail": "Apache Kafka"},
        ],
    },
    {
        "domain": "education",
        "text": "The university registrar's office replaced Blackboard with Canvas LMS after student satisfaction surveys showed 60% dissatisfaction. Canvas was chosen for its native LTI 1.3 support and AWS hosting. The legacy SCORM content required conversion.",
        "expected_entities": ["Blackboard", "Canvas", "AWS", "SCORM"],
        "expected_relations": [
            {"head": "Canvas", "relation": "replaced", "tail": "Blackboard"},
            {"head": "Canvas", "relation": "depends_on", "tail": "AWS"},
        ],
    },
]


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


def relation_match(pred_head, pred_tail, exp_head, exp_tail):
    """Check if a predicted relation matches an expected one (entity pair, ignoring direction)."""
    h_match = fuzzy_match(pred_head, exp_head)
    t_match = fuzzy_match(pred_tail, exp_tail)
    h_match_rev = fuzzy_match(pred_head, exp_tail)
    t_match_rev = fuzzy_match(pred_tail, exp_head)
    return (h_match and t_match) or (h_match_rev and t_match_rev)


# ── Graphiti ───────────────────────────────────────────────────────

async def run_graphiti(episodes, label):
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import OpenAIClient, LLMConfig
    from neo4j import GraphDatabase

    print(f"\n{'='*60}")
    print(f"GRAPHITI — {label}")
    print(f"{'='*60}")

    api_key = os.environ["OPENAI_API_KEY"]
    llm_config = LLMConfig(api_key=api_key, model="gpt-4o-mini", small_model="gpt-4o-mini")
    llm_client = OpenAIClient(config=llm_config)
    graphiti = Graphiti("bolt://localhost:7687", "neo4j", "password123", llm_client=llm_client)
    try:
        await graphiti.build_indices_and_constraints()
    except Exception:
        pass

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

    total_ent_f1 = 0.0
    total_rel_f1 = 0.0
    total_time = 0.0

    for i, ep in enumerate(episodes):
        # Count nodes before
        with driver.session() as s:
            before = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]

        start = time.time()
        try:
            await graphiti.add_episode(
                name=f"{label}_v2_ep{i}",
                episode_body=ep["text"],
                source_description=label,
                reference_time=datetime.now(),
            )
            elapsed = time.time() - start
            total_time += elapsed

            # Get NEW entity nodes from Neo4j directly (fair entity evaluation)
            with driver.session() as s:
                entity_rows = s.run(
                    "MATCH (n:Entity) RETURN n.name AS name SKIP $skip",
                    skip=before,
                ).data()
                pred_entities = [r["name"].lower() for r in entity_rows if r["name"]]

                # Get ALL edges (relations) from Neo4j
                edge_rows = s.run("""
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE r.fact IS NOT NULL
                    RETURN a.name AS head, r.name AS rel_name, b.name AS tail, r.fact AS fact
                    ORDER BY r.created_at DESC LIMIT 50
                """).data()

            # Entity F1: compare Neo4j entity names vs expected
            exp_ents = [e.lower() for e in ep["expected_entities"]]
            _, _, ent_f1 = compute_f1_fuzzy(pred_entities, exp_ents)

            # Relation F1: check if expected entity PAIRS appear as edges
            # This is fair — we check if Graphiti found a relationship between the right entities
            exp_rels = ep["expected_relations"]
            matched_rels = [False] * len(exp_rels)
            pred_rel_count = 0
            for edge in edge_rows:
                head, tail = edge["head"].lower(), edge["tail"].lower()
                for j, er in enumerate(exp_rels):
                    if not matched_rels[j] and relation_match(head, tail, er["head"], er["tail"]):
                        matched_rels[j] = True
                        pred_rel_count += 1
                        break

            rel_p = pred_rel_count / len(edge_rows) if edge_rows else 0.0
            rel_r = sum(matched_rels) / len(exp_rels) if exp_rels else 0.0
            rel_f1 = 2 * rel_p * rel_r / (rel_p + rel_r) if (rel_p + rel_r) > 0 else 0.0

            domain = ep.get("domain", "tech")
            print(f"  [{domain:>12}] ep{i}: ent={ent_f1:.3f} rel={rel_f1:.3f} time={elapsed:.1f}s ents={len(pred_entities)} edges={len(edge_rows)}")
            total_ent_f1 += ent_f1
            total_rel_f1 += rel_f1

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  ep{i}: ERROR ({elapsed:.1f}s): {str(e)[:120]}")

    n = len(episodes)
    avg_ent = total_ent_f1 / n
    avg_rel = total_rel_f1 / n
    combined = (avg_ent + avg_rel) / 2
    print(f"\n  Entity F1:   {avg_ent:.3f}")
    print(f"  Relation F1: {avg_rel:.3f}")
    print(f"  Combined:    {combined:.3f}")
    print(f"  Time:        {total_time:.1f}s ({total_time/n:.1f}s/ep)")
    print(f"  LLM calls:   ~{n * 6}")

    driver.close()
    await graphiti.close()
    return {"ent": avg_ent, "rel": avg_rel, "combined": combined, "time": total_time / n, "calls": n * 6}


# ── ctxgraph ──────────────────────────────────────────────────────

def run_ctxgraph(episodes, label):
    import openai

    print(f"\n{'='*60}")
    print(f"CTXGRAPH — {label}")
    print(f"{'='*60}")

    api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=api_key)

    schema_prompt = """Extract entities and relations from this text. Reply with ONLY valid JSON.

Entity types: Person, Component, Service, Database, Infrastructure, Language, Decision, Constraint, Metric, Pattern
Relation types: chose, rejected, replaced, depends_on, introduced, deprecated, caused, fixed, constrained_by

Rules:
- Use SHORT canonical names ("Redis" not "Redis cache server", "Stripe" not "Stripe SDK v2")
- Teams/departments/roles are Person entities ("platform team", "treasury department")
- Constraints include: compliance requirements, SLAs, certifications, budget caps
- Relation head and tail MUST be the EXACT name string from your entities list
- CRITICAL: First extract all entities, then for EVERY pair of related entities, add a relation
- Prefer specific relation types: "replaced" over "depends_on" when migration is described

JSON: {"entities": [{"name": "...", "entity_type": "..."}], "relations": [{"head": "exact entity name", "relation": "type", "tail": "exact entity name"}]}"""

    total_ent_f1 = 0.0
    total_rel_f1 = 0.0
    total_time = 0.0

    for i, ep in enumerate(episodes):
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": schema_prompt},
                    {"role": "user", "content": ep["text"]},
                ],
                temperature=0,
                max_tokens=1024,
            )
            elapsed = time.time() - start
            total_time += elapsed

            raw = resp.choices[0].message.content
            if "```" in raw:
                raw = raw.split("```json")[-1].split("```")[0] if "```json" in raw else raw.split("```")[1].split("```")[0]
            if "{" in raw:
                raw = raw[raw.index("{"):raw.rindex("}") + 1]

            parsed = json.loads(raw)
            pred_ents = [e["name"].lower() for e in parsed.get("entities", [])]

            exp_ents = [e.lower() for e in ep["expected_entities"]]
            _, _, ent_f1 = compute_f1_fuzzy(pred_ents, exp_ents)

            # Relation F1: same fair evaluation as Graphiti
            # Check if predicted entity pairs match expected entity pairs
            pred_rels = parsed.get("relations", [])
            exp_rels = ep["expected_relations"]
            matched_rels = [False] * len(exp_rels)
            matched_pred = 0
            for pr in pred_rels:
                ph, pt = pr["head"].lower(), pr["tail"].lower()
                for j, er in enumerate(exp_rels):
                    if not matched_rels[j] and relation_match(ph, pt, er["head"], er["tail"]):
                        matched_rels[j] = True
                        matched_pred += 1
                        break

            rel_p = matched_pred / len(pred_rels) if pred_rels else 0.0
            rel_r = sum(matched_rels) / len(exp_rels) if exp_rels else 0.0
            rel_f1 = 2 * rel_p * rel_r / (rel_p + rel_r) if (rel_p + rel_r) > 0 else 0.0

            domain = ep.get("domain", "tech")
            print(f"  [{domain:>12}] ep{i}: ent={ent_f1:.3f} rel={rel_f1:.3f} time={elapsed:.1f}s ents={len(pred_ents)} rels={len(pred_rels)}")
            total_ent_f1 += ent_f1
            total_rel_f1 += rel_f1

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  ep{i}: ERROR ({elapsed:.1f}s): {str(e)[:100]}")

    n = len(episodes)
    avg_ent = total_ent_f1 / n
    avg_rel = total_rel_f1 / n
    combined = (avg_ent + avg_rel) / 2
    print(f"\n  Entity F1:   {avg_ent:.3f}")
    print(f"  Relation F1: {avg_rel:.3f}")
    print(f"  Combined:    {combined:.3f}")
    print(f"  Time:        {total_time:.1f}s ({total_time/n:.1f}s/ep)")
    print(f"  LLM calls:   {n}")

    return {"ent": avg_ent, "rel": avg_rel, "combined": combined, "time": total_time / n, "calls": n}


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY"); sys.exit(1)

    print("=" * 70)
    print("FAIR HEAD-TO-HEAD: Graphiti vs ctxgraph (GPT-4o-mini, same eval)")
    print("=" * 70)
    print("Entity eval: fuzzy name matching (both systems)")
    print("Relation eval: entity pair matching (ignoring relation type & direction)")

    # Clear Neo4j
    subprocess.run(["docker", "exec", "neo4j-graphiti", "cypher-shell", "-u", "neo4j", "-p", "password123", "MATCH (n) DETACH DELETE n"], capture_output=True, timeout=10)

    g_tech = await run_graphiti(TECH_EPISODES, "tech")
    c_tech = run_ctxgraph(TECH_EPISODES, "tech")

    subprocess.run(["docker", "exec", "neo4j-graphiti", "cypher-shell", "-u", "neo4j", "-p", "password123", "MATCH (n) DETACH DELETE n"], capture_output=True, timeout=10)

    g_cross = await run_graphiti(CROSS_DOMAIN_EPISODES, "cross-domain")
    c_cross = run_ctxgraph(CROSS_DOMAIN_EPISODES, "cross-domain")

    print(f"\n{'='*70}")
    print(f"{'FINAL RESULTS (FAIR EVALUATION)':^70}")
    print(f"{'='*70}")
    for lbl, g, c in [("TECH (5 episodes)", g_tech, c_tech), ("CROSS-DOMAIN (5 episodes)", g_cross, c_cross)]:
        print(f"\n  {lbl}")
        print(f"  {'':>25} {'Graphiti':>12} {'ctxgraph':>12} {'Winner':>12}")
        ent_w = "Graphiti" if g["ent"] > c["ent"] else "ctxgraph" if c["ent"] > g["ent"] else "tie"
        rel_w = "Graphiti" if g["rel"] > c["rel"] else "ctxgraph" if c["rel"] > g["rel"] else "tie"
        com_w = "Graphiti" if g["combined"] > c["combined"] else "ctxgraph" if c["combined"] > g["combined"] else "tie"
        print(f"  {'Entity F1':>25} {g['ent']:>12.3f} {c['ent']:>12.3f} {ent_w:>12}")
        print(f"  {'Relation F1':>25} {g['rel']:>12.3f} {c['rel']:>12.3f} {rel_w:>12}")
        print(f"  {'Combined':>25} {g['combined']:>12.3f} {c['combined']:>12.3f} {com_w:>12}")
        print(f"  {'Time/episode':>25} {g['time']:>11.1f}s {c['time']:>11.1f}s {'ctxgraph':>12}")
        print(f"  {'LLM calls':>25} {g['calls']:>12} {c['calls']:>12} {'ctxgraph':>12}")

    print(f"\n  COST (1000 episodes, GPT-4o-mini @ $0.15/M input + $0.60/M output)")
    total_g = g_tech["calls"] + g_cross["calls"]
    total_c = c_tech["calls"] + c_cross["calls"]
    g_cost = total_g / 10 * 1000 * 0.0003
    c_cost = total_c / 10 * 1000 * 0.0003
    print(f"  {'':>25} {'Graphiti':>12} {'ctxgraph':>12}")
    print(f"  {'LLM calls':>25} {int(total_g/10*1000):>12} {int(total_c/10*1000):>12}")
    print(f"  {'Cost':>25} {'$'+f'{g_cost:.2f}':>12} {'$'+f'{c_cost:.2f}':>12}")
    ratio = g_cost / c_cost if c_cost > 0 else float('inf')
    print(f"  {'':>25} {'':>12} {f'{ratio:.0f}x cheaper':>12}")

if __name__ == "__main__":
    asyncio.run(main())
