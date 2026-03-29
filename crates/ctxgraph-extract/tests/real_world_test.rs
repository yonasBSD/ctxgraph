//! Test extraction on real-world tech data: git commits, Sentry errors, PRs, Slack, ADRs.
//!
//! Run: OPENROUTER_API_KEY=... CTXGRAPH_MODELS_DIR=~/.cache/ctxgraph/models \
//!      cargo test --test real_world_test -- --ignored --nocapture

use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Deserialize)]
struct RealWorldEpisode {
    source: String,
    text: String,
    expected_entities: Vec<ExpEntity>,
    expected_relations: Vec<ExpRelation>,
}

#[derive(Debug, Deserialize)]
struct ExpEntity {
    name: String,
    entity_type: String,
}

#[derive(Debug, Deserialize)]
struct ExpRelation {
    head: String,
    relation: String,
    tail: String,
}

fn fuzzy_contains(a: &str, b: &str) -> bool {
    let al = a.to_lowercase();
    let bl = b.to_lowercase();
    al == bl || al.contains(&bl) || bl.contains(&al)
}

fn compute_f1_fuzzy(predicted: &[String], expected: &[String]) -> (f64, f64, f64) {
    if predicted.is_empty() && expected.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    let mut matched = vec![false; expected.len()];
    let mut tp = 0.0;
    for pred in predicted {
        for (i, exp) in expected.iter().enumerate() {
            if !matched[i] && fuzzy_contains(pred, exp) {
                tp += 1.0;
                matched[i] = true;
                break;
            }
        }
    }
    let p = if predicted.is_empty() {
        0.0
    } else {
        tp / predicted.len() as f64
    };
    let r = if expected.is_empty() {
        0.0
    } else {
        tp / expected.len() as f64
    };
    let f1 = if (p + r) == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    };
    (p, r, f1)
}

#[test]
#[ignore]
fn test_real_world_tech_extraction() {
    use chrono::Utc;
    use ctxgraph_extract::pipeline::ExtractionPipeline;
    use ctxgraph_extract::schema::ExtractionSchema;

    let models_dir = std::env::var("CTXGRAPH_MODELS_DIR").unwrap_or_else(|_| {
        dirs::cache_dir()
            .unwrap()
            .join("ctxgraph/models")
            .display()
            .to_string()
    });

    let pipeline = ExtractionPipeline::new(
        ExtractionSchema::default(),
        std::path::Path::new(&models_dir),
        0.2,
    )
    .expect("Failed to create pipeline");

    let data = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/real_world_tech_episodes.json"
    ))
    .unwrap();
    let episodes: Vec<RealWorldEpisode> = serde_json::from_str(&data).unwrap();

    let mut total_ent_f1 = 0.0;
    let mut total_rel_f1 = 0.0;

    for (i, ep) in episodes.iter().enumerate() {
        let result = pipeline.extract(&ep.text, Utc::now()).unwrap();

        let pred_ents: Vec<String> = result
            .entities
            .iter()
            .map(|e| e.text.to_lowercase())
            .collect();
        let exp_ents: Vec<String> = ep
            .expected_entities
            .iter()
            .map(|e| e.name.to_lowercase())
            .collect();
        let (_, _, ent_f1) = compute_f1_fuzzy(&pred_ents, &exp_ents);

        let pred_rels: Vec<String> = result
            .relations
            .iter()
            .map(|r| {
                format!(
                    "{}:{}:{}",
                    r.head.to_lowercase(),
                    r.relation,
                    r.tail.to_lowercase()
                )
            })
            .collect();
        let exp_rels: Vec<String> = ep
            .expected_relations
            .iter()
            .map(|r| {
                format!(
                    "{}:{}:{}",
                    r.head.to_lowercase(),
                    r.relation,
                    r.tail.to_lowercase()
                )
            })
            .collect();
        let (_, _, rel_f1) = compute_f1_fuzzy(&pred_rels, &exp_rels);

        eprintln!(
            "\n[{:>15}] ep{i}: entity={ent_f1:.3} | relation={rel_f1:.3}",
            ep.source
        );
        eprintln!("  Entities found:    {:?}", pred_ents);
        eprintln!("  Entities expected: {:?}", exp_ents);
        if !pred_rels.is_empty() || !exp_rels.is_empty() {
            let pred_set: HashSet<&String> = pred_rels.iter().collect();
            let exp_set: HashSet<&String> = exp_rels.iter().collect();
            let missed: Vec<_> = exp_set.difference(&pred_set).collect();
            let found: Vec<_> = pred_set.intersection(&exp_set).collect();
            let spurious: Vec<_> = pred_set.difference(&exp_set).collect();
            if !found.is_empty() {
                eprintln!("  CORRECT rels: {:?}", found);
            }
            if !missed.is_empty() {
                eprintln!("  MISSED rels:  {:?}", missed);
            }
            if !spurious.is_empty() {
                eprintln!("  SPURIOUS rels: {:?}", spurious);
            }
        }

        total_ent_f1 += ent_f1;
        total_rel_f1 += rel_f1;
    }

    let n = episodes.len() as f64;
    let avg_ent = total_ent_f1 / n;
    let avg_rel = total_rel_f1 / n;
    let combined = (avg_ent + avg_rel) / 2.0;

    eprintln!("\n=== REAL-WORLD TECH RESULTS ===");
    eprintln!("Episodes:            {}", episodes.len());
    eprintln!("Avg entity F1:       {avg_ent:.3}");
    eprintln!("Avg relation F1:     {avg_rel:.3}");
    eprintln!("Combined F1:         {combined:.3}");
    eprintln!("===============================");
}
