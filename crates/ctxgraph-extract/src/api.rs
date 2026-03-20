//! API-based relation extraction using OpenAI or compatible endpoints.
//!
//! Tier 2 extraction: highest quality (~0.85-0.90 F1) but requires an API key
//! and sends data to an external service.
//!
//! Set `CTXGRAPH_API_KEY` to enable. Optionally set `CTXGRAPH_API_URL` and
//! `CTXGRAPH_API_MODEL` to customize the endpoint.

use serde::{Deserialize, Serialize};

use crate::ner::ExtractedEntity;
use crate::rel::ExtractedRelation;
use crate::schema::ExtractionSchema;

const DEFAULT_API_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MODEL: &str = "gpt-4.1-mini";

/// API-based relation extraction engine.
pub struct ApiRelEngine {
    api_url: String,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f64,
    max_tokens: u32,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct LlmTriple {
    head: String,
    relation: String,
    tail: String,
}

impl ApiRelEngine {
    /// Create a new API engine from environment variables.
    /// Returns `None` if `CTXGRAPH_API_KEY` is not set.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("CTXGRAPH_API_KEY").ok()?;
        if api_key.is_empty() {
            return None;
        }
        Some(Self {
            api_url: std::env::var("CTXGRAPH_API_URL")
                .unwrap_or_else(|_| DEFAULT_API_URL.to_string()),
            api_key,
            model: std::env::var("CTXGRAPH_API_MODEL")
                .unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
        })
    }

    /// Extract relations using the API.
    pub fn extract(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        schema: &ExtractionSchema,
    ) -> Result<Vec<ExtractedRelation>, ApiError> {
        let system_prompt = build_system_prompt(schema);
        let user_prompt = build_user_prompt(text, entities, schema);

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: system_prompt,
                },
                Message {
                    role: "user".into(),
                    content: user_prompt,
                },
            ],
            temperature: 0.0,
            max_tokens: 512,
        };

        let response = reqwest::blocking::Client::new()
            .post(&self.api_url)
            .timeout(std::time::Duration::from_secs(30))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| ApiError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().unwrap_or_default();
            return Err(ApiError::HttpStatus(status, body));
        }

        let body = response
            .json::<ChatResponse>()
            .map_err(|e| ApiError::Parse(e.to_string()))?;

        let content = body
            .choices
            .first()
            .map(|c| c.message.content.as_str())
            .unwrap_or("");

        parse_response(content, entities, schema)
    }
}

fn build_system_prompt(schema: &ExtractionSchema) -> String {
    let relation_defs: Vec<String> = schema
        .relation_types
        .iter()
        .map(|(name, spec)| {
            format!(
                "- {name}: {desc} (head: {heads}, tail: {tails})",
                desc = spec.description,
                heads = spec.head.join("/"),
                tails = spec.tail.join("/"),
            )
        })
        .collect();

    format!(
        r#"You are a precise knowledge graph extraction engine. Extract relationships between entities from software architecture discussions.

Relation types:
{relations}

Output ONLY valid JSON lines. Each line: {{"head":"<entity>","relation":"<type>","tail":"<entity>"}}
- For "replaced": head=NEW, tail=OLD (NEW replaced OLD)
- For "depends_on": head=consumer, tail=provider
- Only extract relationships explicitly supported by the text
- Use exact entity names from the provided list"#,
        relations = relation_defs.join("\n"),
    )
}

fn build_user_prompt(
    text: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> String {
    let entity_list: Vec<String> = entities
        .iter()
        .map(|e| format!("{} [{}]", e.text, e.entity_type))
        .collect();

    format!(
        "Entities: {entities}\n\nText: {text}\n\nRelations ({types} only):",
        entities = entity_list.join(", "),
        types = schema.relation_labels().join(", "),
    )
}

fn parse_response(
    content: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> Result<Vec<ExtractedRelation>, ApiError> {
    let relation_names: std::collections::HashSet<&str> =
        schema.relation_labels().into_iter().collect();

    let mut relations = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Try JSON array first
    if let Ok(triples) = serde_json::from_str::<Vec<LlmTriple>>(content.trim()) {
        for t in triples {
            add_triple(&t, entities, &relation_names, &mut seen, &mut relations);
        }
        return Ok(relations);
    }

    // JSON lines
    for line in content.lines() {
        let line = line.trim().trim_end_matches(',');
        if line.is_empty() || !line.starts_with('{') {
            continue;
        }
        if let Ok(triple) = serde_json::from_str::<LlmTriple>(line) {
            add_triple(&triple, entities, &relation_names, &mut seen, &mut relations);
        }
    }

    Ok(relations)
}

fn add_triple(
    triple: &LlmTriple,
    entities: &[ExtractedEntity],
    relation_names: &std::collections::HashSet<&str>,
    seen: &mut std::collections::HashSet<(String, String, String)>,
    relations: &mut Vec<ExtractedRelation>,
) {
    if !relation_names.contains(triple.relation.as_str()) {
        return;
    }

    let head = match_entity(&triple.head, entities);
    let tail = match_entity(&triple.tail, entities);

    let (head_name, tail_name) = match (head, tail) {
        (Some(h), Some(t)) if h.text != t.text => (h.text.clone(), t.text.clone()),
        _ => return,
    };

    let key = (head_name.clone(), triple.relation.clone(), tail_name.clone());
    if !seen.insert(key) {
        return;
    }

    relations.push(ExtractedRelation {
        head: head_name,
        relation: triple.relation.clone(),
        tail: tail_name,
        confidence: 0.85,
    });
}

fn match_entity<'a>(name: &str, entities: &'a [ExtractedEntity]) -> Option<&'a ExtractedEntity> {
    let name_lower = name.to_lowercase();

    // Exact
    if let Some(e) = entities.iter().find(|e| e.text == name) {
        return Some(e);
    }
    // Case-insensitive
    if let Some(e) = entities.iter().find(|e| e.text.to_lowercase() == name_lower) {
        return Some(e);
    }
    // Substring
    entities.iter().find(|e| {
        e.text.to_lowercase().contains(&name_lower)
            || name_lower.contains(&e.text.to_lowercase())
    })
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("API request failed: {0}")]
    Request(String),

    #[error("API returned HTTP {0}: {1}")]
    HttpStatus(u16, String),

    #[error("failed to parse API response: {0}")]
    Parse(String),
}
