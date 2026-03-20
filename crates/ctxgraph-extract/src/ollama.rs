//! Ollama-based relation extraction using local LLMs (Triplex, Qwen, etc.)
//!
//! When Ollama is running locally, this module sends text + extracted entities
//! to a local LLM for relation extraction. Falls back gracefully when Ollama
//! is unavailable.

use serde::{Deserialize, Serialize};

use crate::ner::ExtractedEntity;
use crate::rel::ExtractedRelation;
use crate::schema::ExtractionSchema;

/// Default Ollama API endpoint.
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Default model for relation extraction.
const DEFAULT_MODEL: &str = "sciphi/triplex";

/// Ollama client for LLM-based relation extraction.
pub struct OllamaRelEngine {
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f64,
    num_predict: u32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

/// A triple extracted by the LLM.
#[derive(Debug, Deserialize)]
struct LlmTriple {
    head: String,
    relation: String,
    tail: String,
}

impl OllamaRelEngine {
    /// Create a new Ollama engine with defaults.
    pub fn new() -> Self {
        Self {
            base_url: std::env::var("CTXGRAPH_OLLAMA_URL")
                .unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string()),
            model: std::env::var("CTXGRAPH_OLLAMA_MODEL")
                .unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
        }
    }

    /// Check if Ollama is reachable (fast health check).
    pub fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        reqwest::blocking::Client::new()
            .get(&url)
            .timeout(std::time::Duration::from_millis(500))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Extract relations using the local LLM.
    pub fn extract(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        schema: &ExtractionSchema,
    ) -> Result<Vec<ExtractedRelation>, OllamaError> {
        let prompt = build_prompt(text, entities, schema);

        let request = OllamaRequest {
            model: self.model.clone(),
            prompt,
            stream: false,
            options: OllamaOptions {
                temperature: 0.0,
                num_predict: 512,
            },
        };

        let url = format!("{}/api/generate", self.base_url);
        let response = reqwest::blocking::Client::new()
            .post(&url)
            .timeout(std::time::Duration::from_secs(30))
            .json(&request)
            .send()
            .map_err(|e| OllamaError::Request(e.to_string()))?;

        if !response.status().is_success() {
            return Err(OllamaError::HttpStatus(response.status().as_u16()));
        }

        let body: OllamaResponse = response
            .json::<OllamaResponse>()
            .map_err(|e| OllamaError::Parse(e.to_string()))?;

        parse_llm_response(&body.response, entities, schema)
    }
}

/// Build a Triplex-compatible prompt for relation extraction.
fn build_prompt(
    text: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> String {
    let entity_types: Vec<&str> = schema.entity_labels();
    let relation_types: Vec<String> = schema
        .relation_types
        .iter()
        .map(|(name, spec)| format!("{} ({})", name, spec.description))
        .collect();

    let entity_list: Vec<String> = entities
        .iter()
        .map(|e| format!("- {} [{}]", e.text, e.entity_type))
        .collect();

    format!(
        r#"Extract relationships between entities from the following text.

Entity types: {entity_types}
Relation types: {relation_types}

Already extracted entities:
{entity_list}

Text: {text}

For each relationship found, output one JSON object per line with exactly these fields:
{{"head": "<entity name>", "relation": "<relation type>", "tail": "<entity name>"}}

Rules:
- head and tail MUST be from the extracted entities list above (use exact names)
- relation MUST be one of: {relation_keys}
- For "replaced": head is the NEW thing, tail is the OLD thing (NEW replaced OLD)
- For "depends_on": head is the consumer, tail is the provider
- Only output relationships supported by the text. Do not hallucinate.
- Output ONLY JSON lines, no other text.

Relationships:"#,
        entity_types = entity_types.join(", "),
        relation_types = relation_types.join(", "),
        entity_list = entity_list.join("\n"),
        relation_keys = schema.relation_labels().join(", "),
    )
}

/// Parse the LLM response into ExtractedRelation structs.
///
/// Accepts JSON lines, JSON array, or mixed text with embedded JSON objects.
fn parse_llm_response(
    response: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> Result<Vec<ExtractedRelation>, OllamaError> {
    let entity_names: std::collections::HashSet<String> = entities
        .iter()
        .map(|e| e.text.to_lowercase())
        .collect();
    let relation_names: std::collections::HashSet<&str> = schema
        .relation_labels()
        .into_iter()
        .collect();

    let mut relations = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Try parsing as JSON array first
    if let Ok(triples) = serde_json::from_str::<Vec<LlmTriple>>(response.trim()) {
        for t in triples {
            if validate_and_add(&t, &entity_names, &relation_names, &mut seen, &mut relations, entities) {
                continue;
            }
        }
        return Ok(relations);
    }

    // Fall back to JSON-lines parsing
    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with('{') {
            continue;
        }

        // Strip trailing comma if present
        let json_str = line.trim_end_matches(',');

        if let Ok(triple) = serde_json::from_str::<LlmTriple>(json_str) {
            validate_and_add(&triple, &entity_names, &relation_names, &mut seen, &mut relations, entities);
        }
    }

    Ok(relations)
}

/// Validate a triple against known entities/relations and add it if valid.
/// Returns true if the triple was valid (whether or not it was a duplicate).
fn validate_and_add(
    triple: &LlmTriple,
    _entity_names: &std::collections::HashSet<String>,
    relation_names: &std::collections::HashSet<&str>,
    seen: &mut std::collections::HashSet<(String, String, String)>,
    relations: &mut Vec<ExtractedRelation>,
    entities: &[ExtractedEntity],
) -> bool {
    // Validate relation type
    if !relation_names.contains(triple.relation.as_str()) {
        return false;
    }

    // Match head and tail to known entities (case-insensitive fuzzy match)
    let head = match_entity(&triple.head, entities);
    let tail = match_entity(&triple.tail, entities);

    let (head_name, tail_name) = match (head, tail) {
        (Some(h), Some(t)) if h.text != t.text => (h.text.clone(), t.text.clone()),
        _ => return false,
    };

    let key = (head_name.clone(), triple.relation.clone(), tail_name.clone());
    if !seen.insert(key) {
        return true; // duplicate but valid
    }

    relations.push(ExtractedRelation {
        head: head_name,
        relation: triple.relation.clone(),
        tail: tail_name,
        confidence: 0.70, // LLM-based extraction gets moderate confidence
    });
    true
}

/// Find the best matching entity for an LLM-produced name.
///
/// Tries exact match first, then case-insensitive, then substring.
fn match_entity<'a>(name: &str, entities: &'a [ExtractedEntity]) -> Option<&'a ExtractedEntity> {
    let name_lower = name.to_lowercase();

    // Exact match
    if let Some(e) = entities.iter().find(|e| e.text == name) {
        return Some(e);
    }

    // Case-insensitive
    if let Some(e) = entities.iter().find(|e| e.text.to_lowercase() == name_lower) {
        return Some(e);
    }

    // Substring match (LLM might say "PostgreSQL" when entity is "PostgreSQL 14")
    if let Some(e) = entities.iter().find(|e| {
        e.text.to_lowercase().contains(&name_lower)
            || name_lower.contains(&e.text.to_lowercase())
    }) {
        return Some(e);
    }

    None
}

#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("Ollama request failed: {0}")]
    Request(String),

    #[error("Ollama returned HTTP {0}")]
    HttpStatus(u16),

    #[error("failed to parse Ollama response: {0}")]
    Parse(String),
}
