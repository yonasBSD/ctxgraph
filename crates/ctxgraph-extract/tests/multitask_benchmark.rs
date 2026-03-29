//! Benchmark the GLiNER multitask model (gline-rs) for joint NER + relation extraction.
//!
//! This tests the ModelBasedRelEngine directly against both tech and cross-domain episodes.
//! Run: CTXGRAPH_MODELS_DIR=~/.cache/ctxgraph/models cargo test --test multitask_benchmark -- --ignored --nocapture

use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct BenchmarkEpisode {
    text: String,
    expected_entities: Vec<ExpectedEntity>,
    expected_relations: Vec<ExpectedRelation>,
}

#[derive(Debug, Deserialize)]
struct ExpectedEntity {
    name: String,
    entity_type: String,
    #[allow(dead_code)]
    span_start: Option<usize>,
    #[allow(dead_code)]
    span_end: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ExpectedRelation {
    head: String,
    relation: String,
    tail: String,
}

fn compute_f1(predicted: &[String], expected: &[String]) -> (f64, f64, f64) {
    if predicted.is_empty() && expected.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    let pred_set: HashSet<&String> = predicted.iter().collect();
    let exp_set: HashSet<&String> = expected.iter().collect();
    let tp = pred_set.intersection(&exp_set).count() as f64;
    let p = if pred_set.is_empty() {
        0.0
    } else {
        tp / pred_set.len() as f64
    };
    let r = if exp_set.is_empty() {
        0.0
    } else {
        tp / exp_set.len() as f64
    };
    let f1 = if (p + r) == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    };
    (p, r, f1)
}

fn load_tech_episodes() -> Vec<BenchmarkEpisode> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/benchmark_episodes.json"
    );
    let data = std::fs::read_to_string(path).expect("Failed to read benchmark_episodes.json");
    serde_json::from_str(&data).expect("Failed to deserialize")
}

fn load_cross_domain_episodes() -> Vec<(String, BenchmarkEpisode)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/cross_domain_episodes.json"
    );
    let data = std::fs::read_to_string(path).expect("Failed to read cross_domain_episodes.json");

    #[derive(Deserialize)]
    struct CrossDomainEpisode {
        domain: String,
        text: String,
        expected_entities: Vec<ExpectedEntity>,
        expected_relations: Vec<ExpectedRelation>,
    }

    let episodes: Vec<CrossDomainEpisode> = serde_json::from_str(&data).unwrap();
    episodes
        .into_iter()
        .map(|e| {
            (
                e.domain,
                BenchmarkEpisode {
                    text: e.text,
                    expected_entities: e.expected_entities,
                    expected_relations: e.expected_relations,
                },
            )
        })
        .collect()
}

/// Benchmark: GLiNER multitask model (joint NER + RE) on tech episodes.
#[test]
#[ignore]
fn test_multitask_tech_benchmark() {
    use ctxgraph_extract::rel::ModelBasedRelEngine;
    use ctxgraph_extract::schema::ExtractionSchema;

    let models_dir = std::env::var("CTXGRAPH_MODELS_DIR").unwrap_or_else(|_| {
        dirs::cache_dir()
            .expect("no cache dir")
            .join("ctxgraph/models")
            .display()
            .to_string()
    });

    let model_path =
        Path::new(&models_dir).join("gliner-multitask-large-v0.5/onnx/model_int8.onnx");
    let tokenizer_path = Path::new(&models_dir).join("gliner-multitask-large-v0.5/tokenizer.json");

    if !model_path.exists() {
        eprintln!("Multitask model not found at {}", model_path.display());
        return;
    }

    let engine =
        ModelBasedRelEngine::new(&model_path, &tokenizer_path).expect("Failed to load multitask");
    let schema = ExtractionSchema::default();
    // Use entity key names (Person, Component, etc.) not descriptions
    let labels: Vec<&str> = schema.entity_labels();

    let episodes = load_tech_episodes();
    let mut total_entity_f1 = 0.0;
    let mut total_relation_f1 = 0.0;

    for (i, ep) in episodes.iter().enumerate() {
        let result = engine.extract(&ep.text, &labels, &schema);
        let (entities, relations) = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Episode {i}: extraction error: {e}");
                (vec![], vec![])
            }
        };

        let pred_ents: Vec<String> = entities
            .iter()
            .map(|e| format!("{}:{}", e.text, e.entity_type))
            .collect();
        let exp_ents: Vec<String> = ep
            .expected_entities
            .iter()
            .map(|e| format!("{}:{}", e.name, e.entity_type))
            .collect();
        let (_, _, ent_f1) = compute_f1(&pred_ents, &exp_ents);

        let pred_rels: Vec<String> = relations
            .iter()
            .map(|r| format!("{}:{}:{}", r.head, r.relation, r.tail))
            .collect();
        let exp_rels: Vec<String> = ep
            .expected_relations
            .iter()
            .map(|r| format!("{}:{}:{}", r.head, r.relation, r.tail))
            .collect();
        let (_, _, rel_f1) = compute_f1(&pred_rels, &exp_rels);

        eprintln!(
            "Episode {i:2}: entity={ent_f1:.3} | relation={rel_f1:.3}  entities={} rels={}",
            entities.len(),
            relations.len()
        );

        if rel_f1 < 1.0 && !exp_rels.is_empty() {
            let pred_set: HashSet<&String> = pred_rels.iter().collect();
            let exp_set: HashSet<&String> = exp_rels.iter().collect();
            let missed: Vec<_> = exp_set.difference(&pred_set).collect();
            let spurious: Vec<_> = pred_set.difference(&exp_set).collect();
            if !missed.is_empty() {
                eprintln!("  MISSED: {:?}", missed);
            }
            if !spurious.is_empty() {
                eprintln!("  SPURIOUS: {:?}", spurious);
            }
        }

        total_entity_f1 += ent_f1;
        total_relation_f1 += rel_f1;
    }

    let n = episodes.len() as f64;
    let avg_ent = total_entity_f1 / n;
    let avg_rel = total_relation_f1 / n;
    let combined = (avg_ent + avg_rel) / 2.0;

    eprintln!();
    eprintln!("=== MULTITASK TECH BENCHMARK ===");
    eprintln!("Average entity F1:   {avg_ent:.3}");
    eprintln!("Average relation F1: {avg_rel:.3}");
    eprintln!("Combined F1:         {combined:.3}");
    eprintln!("================================");
}

/// Benchmark: GLiNER multitask model (joint NER + RE) on cross-domain episodes.
#[test]
#[ignore]
fn test_multitask_cross_domain_benchmark() {
    use ctxgraph_extract::rel::ModelBasedRelEngine;
    use ctxgraph_extract::schema::ExtractionSchema;

    let models_dir = std::env::var("CTXGRAPH_MODELS_DIR").unwrap_or_else(|_| {
        dirs::cache_dir()
            .expect("no cache dir")
            .join("ctxgraph/models")
            .display()
            .to_string()
    });

    let model_path =
        Path::new(&models_dir).join("gliner-multitask-large-v0.5/onnx/model_int8.onnx");
    let tokenizer_path = Path::new(&models_dir).join("gliner-multitask-large-v0.5/tokenizer.json");

    if !model_path.exists() {
        eprintln!("Multitask model not found at {}", model_path.display());
        return;
    }

    let engine =
        ModelBasedRelEngine::new(&model_path, &tokenizer_path).expect("Failed to load multitask");
    let schema = ExtractionSchema::default();
    let labels: Vec<&str> = schema.entity_labels();

    let episodes = load_cross_domain_episodes();
    let mut domain_scores: std::collections::BTreeMap<String, Vec<(f64, f64)>> =
        std::collections::BTreeMap::new();

    for (domain, ep) in &episodes {
        let result = engine.extract(&ep.text, &labels, &schema);
        let (entities, relations) = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[{domain:>15}] extraction error: {e}");
                (vec![], vec![])
            }
        };

        let pred_ents: Vec<String> = entities
            .iter()
            .map(|e| format!("{}:{}", e.text, e.entity_type))
            .collect();
        let exp_ents: Vec<String> = ep
            .expected_entities
            .iter()
            .map(|e| format!("{}:{}", e.name, e.entity_type))
            .collect();
        let (_, _, ent_f1) = compute_f1(&pred_ents, &exp_ents);

        let pred_rels: Vec<String> = relations
            .iter()
            .map(|r| format!("{}:{}:{}", r.head, r.relation, r.tail))
            .collect();
        let exp_rels: Vec<String> = ep
            .expected_relations
            .iter()
            .map(|r| format!("{}:{}:{}", r.head, r.relation, r.tail))
            .collect();
        let (_, _, rel_f1) = compute_f1(&pred_rels, &exp_rels);

        eprintln!(
            "[{domain:>15}] entity={ent_f1:.3} | relation={rel_f1:.3}  entities={} rels={}",
            entities.len(),
            relations.len()
        );

        if !pred_rels.is_empty() || !exp_rels.is_empty() {
            let pred_set: HashSet<&String> = pred_rels.iter().collect();
            let exp_set: HashSet<&String> = exp_rels.iter().collect();
            let missed: Vec<_> = exp_set.difference(&pred_set).collect();
            if !missed.is_empty() {
                eprintln!("  MISSED: {:?}", missed);
            }
            if !pred_rels.is_empty() {
                eprintln!("  FOUND:  {:?}", pred_rels);
            }
        }

        domain_scores
            .entry(domain.clone())
            .or_default()
            .push((ent_f1, rel_f1));
    }

    eprintln!();
    eprintln!("=== MULTITASK CROSS-DOMAIN RESULTS ===");
    let mut total_ent = 0.0;
    let mut total_rel = 0.0;
    let mut total_n = 0;
    for (domain, scores) in &domain_scores {
        let avg_ent: f64 = scores.iter().map(|(e, _)| e).sum::<f64>() / scores.len() as f64;
        let avg_rel: f64 = scores.iter().map(|(_, r)| r).sum::<f64>() / scores.len() as f64;
        let combined = (avg_ent + avg_rel) / 2.0;
        eprintln!(
            "{domain:>15}: entity={avg_ent:.3}  relation={avg_rel:.3}  combined={combined:.3}  (n={})",
            scores.len()
        );
        total_ent += scores.iter().map(|(e, _)| e).sum::<f64>();
        total_rel += scores.iter().map(|(_, r)| r).sum::<f64>();
        total_n += scores.len();
    }
    let avg_ent = total_ent / total_n as f64;
    let avg_rel = total_rel / total_n as f64;
    let combined = (avg_ent + avg_rel) / 2.0;
    eprintln!();
    eprintln!("Overall avg entity F1:    {avg_ent:.3}");
    eprintln!("Overall avg relation F1:  {avg_rel:.3}");
    eprintln!("Overall combined F1:      {combined:.3}");
    eprintln!("======================================");
}
