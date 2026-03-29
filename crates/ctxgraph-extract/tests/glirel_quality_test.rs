use ctxgraph_extract::glirel::GlirelEngine;
use ctxgraph_extract::ner::ExtractedEntity;
use std::path::Path;

#[test]
#[ignore]
fn test_glirel_standalone_quality() {
    let dir = std::env::var("CTXGRAPH_MODELS_DIR").unwrap_or_else(|_| {
        dirs::cache_dir()
            .unwrap()
            .join("ctxgraph/models")
            .display()
            .to_string()
    });
    let glirel_dir = Path::new(&dir).join("glirel-large-v0");
    let engine = GlirelEngine::new(&glirel_dir).expect("load glirel");

    let labels = &[
        "chose",
        "rejected",
        "replaced",
        "depends_on",
        "introduced",
        "deprecated",
        "caused",
        "fixed",
        "constrained_by",
    ];

    let cases: Vec<(&str, Vec<ExtractedEntity>)> = vec![
        (
            "Alice chose PostgreSQL over MySQL for the billing service",
            vec![
                ent("Alice", "Person", 0, 5),
                ent("PostgreSQL", "Database", 12, 22),
                ent("MySQL", "Database", 28, 33),
            ],
        ),
        (
            "The risk management team evaluated Bloomberg Terminal and rejected Refinitiv Eikon due to the budget cap",
            vec![
                ent("Bloomberg Terminal", "Component", 41, 59),
                ent("Refinitiv Eikon", "Component", 73, 88),
            ],
        ),
        (
            "The radiology department chose MONAI over TensorFlow for medical imaging constrained by HIPAA",
            vec![
                ent("MONAI", "Component", 31, 36),
                ent("TensorFlow", "Component", 42, 52),
                ent("HIPAA", "Constraint", 89, 94),
            ],
        ),
        (
            "The litigation team replaced Relativity with Everlaw for e-discovery citing FedRAMP as a requirement",
            vec![
                ent("Relativity", "Component", 29, 39),
                ent("Everlaw", "Component", 45, 52),
                ent("FedRAMP", "Constraint", 76, 83),
            ],
        ),
        (
            "The plant manager chose Siemens MindSphere over PTC ThingWorx for predictive maintenance",
            vec![
                ent("Siemens MindSphere", "Component", 24, 42),
                ent("PTC ThingWorx", "Component", 48, 61),
            ],
        ),
        (
            "We migrated from Slack to Microsoft Teams because of the enterprise licensing bundle",
            vec![
                ent("Slack", "Service", 18, 23),
                ent("Microsoft Teams", "Service", 27, 42),
            ],
        ),
        (
            "Kotlin replaced COBOL in the clearing system and fixed the T+3 settlement bottleneck",
            vec![
                ent("Kotlin", "Language", 0, 6),
                ent("COBOL", "Language", 16, 21),
            ],
        ),
        (
            "The CTO deprecated Cerner after the ransomware incident and introduced Oracle Health Cloud",
            vec![
                ent("Cerner", "Service", 19, 25),
                ent("Oracle Health Cloud", "Service", 71, 90),
            ],
        ),
    ];

    for (text, entities) in &cases {
        eprintln!("\n=== {} ===", &text[..text.len().min(70)]);

        for threshold in [0.1, 0.3, 0.5] {
            let rels = engine.extract(text, entities, labels, threshold).unwrap();
            eprintln!("  threshold={threshold:.1}: {} relations", rels.len());
            for r in &rels {
                eprintln!(
                    "    {} --[{} ({:.3})]-> {}",
                    r.head, r.relation, r.confidence, r.tail
                );
            }
        }
    }
}

fn ent(text: &str, typ: &str, start: usize, end: usize) -> ExtractedEntity {
    ExtractedEntity {
        text: text.into(),
        entity_type: typ.into(),
        span_start: start,
        span_end: end,
        confidence: 0.9,
    }
}
