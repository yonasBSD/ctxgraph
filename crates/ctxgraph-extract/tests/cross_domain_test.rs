use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Deserialize)]
struct CrossDomainEpisode {
    domain: String,
    text: String,
    expected_entities: Vec<ExpectedEntity>,
    expected_relations: Vec<ExpectedRelation>,
}

#[derive(Debug, Deserialize)]
struct ExpectedEntity {
    name: String,
    entity_type: String,
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

    let predicted_set: HashSet<&String> = predicted.iter().collect();
    let expected_set: HashSet<&String> = expected.iter().collect();

    let true_positives = predicted_set.intersection(&expected_set).count() as f64;

    let precision = if predicted_set.is_empty() {
        0.0
    } else {
        true_positives / predicted_set.len() as f64
    };

    let recall = if expected_set.is_empty() {
        0.0
    } else {
        true_positives / expected_set.len() as f64
    };

    let f1 = if (precision + recall) == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    (precision, recall, f1)
}

fn load_cross_domain_episodes() -> Vec<CrossDomainEpisode> {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/cross_domain_episodes.json"
    );
    let data =
        std::fs::read_to_string(fixture_path).expect("Failed to read cross_domain_episodes.json");
    serde_json::from_str(&data).expect("Failed to deserialize cross-domain episodes")
}

/// Run extraction against finance, healthcare, legal, manufacturing, education,
/// and government episodes to validate cross-domain generalization.
///
/// Run with:
///   CTXGRAPH_MODELS_DIR=~/.cache/ctxgraph/models \
///     cargo test --package ctxgraph-extract --test cross_domain_test -- --ignored --nocapture
#[test]
#[ignore]
fn test_cross_domain_extraction() {
    use chrono::Utc;
    use ctxgraph_extract::pipeline::ExtractionPipeline;
    use ctxgraph_extract::schema::ExtractionSchema;

    let models_dir = std::env::var("CTXGRAPH_MODELS_DIR").unwrap_or_else(|_| {
        let home = dirs::cache_dir().expect("no cache dir");
        home.join("ctxgraph").join("models").display().to_string()
    });

    let pipeline = ExtractionPipeline::new(
        ExtractionSchema::default(),
        std::path::Path::new(&models_dir),
        0.2,
    )
    .expect("Failed to create pipeline. Are models downloaded?");

    let episodes = load_cross_domain_episodes();

    let mut domain_scores: std::collections::BTreeMap<String, Vec<(f64, f64)>> =
        std::collections::BTreeMap::new();

    let mut total_entity_f1 = 0.0;
    let mut total_entity_text_f1 = 0.0;
    let mut total_relation_f1 = 0.0;

    for (i, ep) in episodes.iter().enumerate() {
        let result = pipeline
            .extract(&ep.text, Utc::now())
            .unwrap_or_else(|e| panic!("Extraction failed on episode {i} ({}): {e}", ep.domain));

        // Entity F1 (name only, case-insensitive)
        let predicted_entities: Vec<String> = result
            .entities
            .iter()
            .map(|e| e.text.to_lowercase())
            .collect();
        let expected_entities: Vec<String> = ep
            .expected_entities
            .iter()
            .map(|e| e.name.to_lowercase())
            .collect();
        let (ep_p, ep_r, entity_f1) = compute_f1(&predicted_entities, &expected_entities);

        // Entity F1 (name + type, strict)
        let predicted_strict: Vec<String> = result
            .entities
            .iter()
            .map(|e| format!("{}:{}", e.text.to_lowercase(), e.entity_type))
            .collect();
        let expected_strict: Vec<String> = ep
            .expected_entities
            .iter()
            .map(|e| format!("{}:{}", e.name.to_lowercase(), e.entity_type))
            .collect();
        let (_, _, entity_strict_f1) = compute_f1(&predicted_strict, &expected_strict);

        // Relation F1 (head:relation:tail, case-insensitive)
        let predicted_relations: Vec<String> = result
            .relations
            .iter()
            .map(|r| {
                format!(
                    "{}:{}:{}",
                    r.head.to_lowercase(),
                    r.relation.to_lowercase(),
                    r.tail.to_lowercase()
                )
            })
            .collect();
        let expected_relations: Vec<String> = ep
            .expected_relations
            .iter()
            .map(|r| {
                format!(
                    "{}:{}:{}",
                    r.head.to_lowercase(),
                    r.relation.to_lowercase(),
                    r.tail.to_lowercase()
                )
            })
            .collect();
        let (rp_p, rp_r, relation_f1) = compute_f1(&predicted_relations, &expected_relations);

        eprintln!(
            "[{:>15}] ep{i:2}: entity={entity_f1:.3} (P={ep_p:.3} R={ep_r:.3}) strict={entity_strict_f1:.3} | rel={relation_f1:.3} (P={rp_p:.3} R={rp_r:.3})",
            ep.domain,
        );

        // Show what was extracted vs expected
        eprintln!("  Entities found: {:?}", predicted_entities);
        eprintln!("  Entities expected: {:?}", expected_entities);

        if relation_f1 < 1.0 {
            let pred_set: HashSet<&String> = predicted_relations.iter().collect();
            let exp_set: HashSet<&String> = expected_relations.iter().collect();
            let missed: Vec<&&String> = exp_set.difference(&pred_set).collect();
            let spurious: Vec<&&String> = pred_set.difference(&exp_set).collect();
            if !missed.is_empty() {
                eprintln!("  MISSED rels: {:?}", missed);
            }
            if !spurious.is_empty() {
                eprintln!("  SPURIOUS rels: {:?}", spurious);
            }
        }
        eprintln!();

        total_entity_f1 += entity_f1;
        total_entity_text_f1 += entity_strict_f1;
        total_relation_f1 += relation_f1;

        domain_scores
            .entry(ep.domain.clone())
            .or_default()
            .push((entity_f1, relation_f1));
    }

    let n = episodes.len() as f64;
    let avg_entity_f1 = total_entity_f1 / n;
    let avg_relation_f1 = total_relation_f1 / n;
    let combined_f1 = (avg_entity_f1 + avg_relation_f1) / 2.0;

    eprintln!();
    eprintln!("=== CROSS-DOMAIN RESULTS ===");
    for (domain, scores) in &domain_scores {
        let dn = scores.len() as f64;
        let avg_e: f64 = scores.iter().map(|(e, _)| e).sum::<f64>() / dn;
        let avg_r: f64 = scores.iter().map(|(_, r)| r).sum::<f64>() / dn;
        eprintln!(
            "  {domain:>15}: entity={avg_e:.3}  relation={avg_r:.3}  combined={:.3}  (n={dn:.0})",
            (avg_e + avg_r) / 2.0
        );
    }
    eprintln!();
    eprintln!("Overall avg entity F1:    {avg_entity_f1:.3}");
    eprintln!("Overall avg relation F1:  {avg_relation_f1:.3}");
    eprintln!("Overall combined F1:      {combined_f1:.3}");
    eprintln!("============================");

    // We don't assert a threshold here — this is exploratory.
    // But we log everything so you can see where it breaks.
}
