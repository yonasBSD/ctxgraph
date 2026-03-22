//! API-based relation extraction using OpenAI, Anthropic, or compatible endpoints.
//!
//! Tier 2 extraction: highest quality (~0.85-0.90 F1) but requires an API key
//! and sends data to an external service.
//!
//! Set `CTXGRAPH_API_KEY` to enable. Optionally set `CTXGRAPH_API_URL` and
//! `CTXGRAPH_API_MODEL` to customize the endpoint.
//!
//! Anthropic detection: if `CTXGRAPH_API_URL` contains "anthropic.com" or
//! `CTXGRAPH_API_MODEL` starts with "claude-", the Anthropic Messages API
//! format is used automatically.

use serde::{Deserialize, Serialize};

use crate::ner::ExtractedEntity;
use crate::rel::ExtractedRelation;
use crate::schema::ExtractionSchema;

const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4o";
const DEFAULT_ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-haiku-4-5-20251001";

/// Which API provider format to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApiProvider {
    OpenAI,
    Anthropic,
}

/// API-based relation extraction engine.
pub struct ApiRelEngine {
    api_url: String,
    api_key: String,
    model: String,
    provider: ApiProvider,
}

// --- OpenAI request/response types ---

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

// --- Anthropic request/response types ---

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    temperature: f64,
    system: String,
    messages: Vec<AnthropicMessage>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LlmTriple {
    head: String,
    relation: String,
    tail: String,
}

/// Detect provider from URL and model name.
fn detect_provider(api_url: &str, model: &str) -> ApiProvider {
    if api_url.contains("anthropic.com") || model.starts_with("claude-") {
        ApiProvider::Anthropic
    } else {
        ApiProvider::OpenAI
    }
}

impl ApiRelEngine {
    /// Create a new API engine from environment variables.
    /// Returns `None` if `CTXGRAPH_API_KEY` is not set.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("CTXGRAPH_API_KEY").ok()?;
        if api_key.is_empty() {
            return None;
        }

        let explicit_url = std::env::var("CTXGRAPH_API_URL").ok();
        let explicit_model = std::env::var("CTXGRAPH_API_MODEL").ok();

        // Detect provider from whatever hints are available.
        let provider = detect_provider(
            explicit_url.as_deref().unwrap_or(""),
            explicit_model.as_deref().unwrap_or(""),
        );

        let (default_url, default_model) = match provider {
            ApiProvider::Anthropic => (DEFAULT_ANTHROPIC_URL, DEFAULT_ANTHROPIC_MODEL),
            ApiProvider::OpenAI => (DEFAULT_OPENAI_URL, DEFAULT_OPENAI_MODEL),
        };

        Some(Self {
            api_url: explicit_url.unwrap_or_else(|| default_url.to_string()),
            api_key,
            model: explicit_model.unwrap_or_else(|| default_model.to_string()),
            provider,
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

        let content = match self.provider {
            ApiProvider::Anthropic => self.call_anthropic(&system_prompt, &user_prompt)?,
            ApiProvider::OpenAI => self.call_openai(&system_prompt, &user_prompt)?,
        };

        parse_response(&content, entities, schema)
    }

    /// Send request using OpenAI chat completions format.
    fn call_openai(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, ApiError> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: system_prompt.to_string(),
                },
                Message {
                    role: "user".into(),
                    content: user_prompt.to_string(),
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

        Ok(body
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default())
    }

    /// Send request using Anthropic Messages API format.
    fn call_anthropic(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, ApiError> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 512,
            temperature: 0.0,
            system: system_prompt.to_string(),
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: user_prompt.to_string(),
            }],
        };

        let response = reqwest::blocking::Client::new()
            .post(&self.api_url)
            .timeout(std::time::Duration::from_secs(30))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
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
            .json::<AnthropicResponse>()
            .map_err(|e| ApiError::Parse(e.to_string()))?;

        // Extract text from the first text content block.
        Ok(body
            .content
            .iter()
            .find(|c| c.content_type == "text")
            .and_then(|c| c.text.clone())
            .unwrap_or_default())
    }
}

fn build_system_prompt(schema: &ExtractionSchema) -> String {
    format!(
        r#"Extract the 2-3 most important directed relationships from software architecture text. Be precise — only extract relationships clearly stated in the text.

Relation types (head → tail):
- chose: person → technology_adopted. "Alice chose PostgreSQL" → chose(Alice,PostgreSQL)
- rejected: person → technology_rejected
- replaced: new → old. "from X to Y" → replaced(Y,X). Don't emit for version upgrades.
- depends_on: consumer → provider. "uses", "backed by", "managed by", "runs on"
- fixed: fixer → thing_fixed
- introduced: actor → new_capability. "added Prometheus" → introduced(Service,Prometheus)
- deprecated: actor → removed_thing
- caused: cause → effect. Metric changes only.
- constrained_by: thing → constraint/SLA/policy

Examples:

Text: "Add Prometheus metrics to the BillingService. CPU utilization and request latency are now scraped every 15 seconds by Grafana dashboards."
Output: [{{"head":"BillingService","relation":"introduced","tail":"Prometheus"}},{{"head":"Grafana","relation":"depends_on","tail":"Prometheus"}}]

Text: "Upgrade the RecommendationService from Java 11 to Java 21 for virtual threads support. GC pause times dropped to sub-millisecond."
Output: [{{"head":"RecommendationService","relation":"introduced","tail":"virtual threads"}},{{"head":"virtual threads","relation":"caused","tail":"GC pause times"}}]

Text: "Enable TLS termination at the Nginx ingress controller. All traffic between the API Gateway and backend services is now encrypted."
Output: [{{"head":"Nginx","relation":"introduced","tail":"TLS"}},{{"head":"API Gateway","relation":"depends_on","tail":"Nginx"}}]

Text: "chore: Jack migrated the LogAggregator from Fluentd to Vector for better throughput and lower resource usage"
Output: [{{"head":"Jack","relation":"chose","tail":"Vector"}},{{"head":"Vector","relation":"replaced","tail":"Fluentd"}},{{"head":"LogAggregator","relation":"depends_on","tail":"Vector"}}]

Rules:
1. Use ONLY entity names from the provided list — copy exactly.
2. Extract 2-3 relationships maximum. Precision over recall.
3. Only use: {relation_keys}"#,
        relation_keys = schema.relation_labels().join(", "),
    )
}

fn build_user_prompt(
    text: &str,
    entities: &[ExtractedEntity],
    _schema: &ExtractionSchema,
) -> String {
    let entity_names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();

    format!(
        r#"Entities: {entity_list}

Text: {text}

Output ONLY a JSON array."#,
        entity_list = entity_names.join(", "),
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

// ---------------------------------------------------------------------------
// Entity cleanup via API
// ---------------------------------------------------------------------------

impl ApiRelEngine {
    /// Clean up entity names using the API LLM.
    ///
    /// Sends GLiNER's entity extractions to GPT-4.1-mini (or configured model)
    /// to fix verbose names and remove generic non-entities.
    ///
    /// Returns a mapping from original entity text → cleaned entity text.
    /// Entities not in the result should be removed (LLM deemed them generic).
    pub fn cleanup_entities(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> Result<std::collections::HashMap<String, String>, ApiError> {
        if entities.is_empty() {
            return Ok(std::collections::HashMap::new());
        }

        let entity_names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        let entity_json = serde_json::to_string(&entity_names)
            .map_err(|e| ApiError::Parse(e.to_string()))?;

        let system = "You are a precise NER post-processor. Fix entity names extracted by a NER model.";
        let user = format!(
            r#"A NER model extracted these entities from the text below. Some names may have extra words or be incorrect. Fix them.

Extracted entities: {entity_json}

Text: {text}

Rules:
- Fix entity names to be the shortest precise form (e.g., "Terraform modules" → "Terraform", "IAM roles" → "IAM")
- Remove generic entities that aren't real names (e.g., "services", "resources", "modules")
- Keep entities that are correct as-is
- Return ONLY a JSON array of corrected entity name strings

Output ONLY a JSON array of strings, no other text."#
        );

        let content = match self.provider {
            ApiProvider::Anthropic => self.call_anthropic(system, &user)?,
            ApiProvider::OpenAI => self.call_openai(system, &user)?,
        };

        parse_cleanup_response(&content, &entity_names)
    }
}

/// Parse the API cleanup response into an original→cleaned name mapping.
fn parse_cleanup_response(
    response: &str,
    originals: &[&str],
) -> Result<std::collections::HashMap<String, String>, ApiError> {
    let mut mapping = std::collections::HashMap::new();

    // Extract JSON array from response
    let text = response.trim();
    let json_str = if text.starts_with('[') {
        text.to_string()
    } else {
        // Try markdown code block
        let re = regex::Regex::new(r"```(?:json)?\s*\n?(.*?)\n?```").unwrap();
        re.captures(text)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_else(|| text.to_string())
    };

    let cleaned: Vec<String> = serde_json::from_str(&json_str)
        .map_err(|e| ApiError::Parse(format!("cleanup JSON: {}", e)))?;

    // Match cleaned names back to originals
    let mut used_originals = std::collections::HashSet::new();

    for cleaned_name in &cleaned {
        let cleaned_lower = cleaned_name.to_lowercase();

        // Exact match
        if let Some(&orig) = originals.iter().find(|&&o| {
            o.to_lowercase() == cleaned_lower && !used_originals.contains(o)
        }) {
            mapping.insert(orig.to_string(), cleaned_name.clone());
            used_originals.insert(orig);
            continue;
        }

        // Substring match (original contains cleaned or vice versa)
        if let Some(&orig) = originals.iter().find(|&&o| {
            !used_originals.contains(o)
                && (o.to_lowercase().contains(&cleaned_lower)
                    || cleaned_lower.contains(&o.to_lowercase()))
        }) {
            mapping.insert(orig.to_string(), cleaned_name.clone());
            used_originals.insert(orig);
        }
    }

    Ok(mapping)
}

/// Apply API-based entity cleanup to extracted entities.
///
/// Only processes multi-word entities and known generic single-word entities.
/// Single-word proper nouns are never sent to the API.
pub fn api_cleanup_entities(
    text: &str,
    entities: &mut Vec<ExtractedEntity>,
) {
    if std::env::var("CTXGRAPH_NO_API").is_ok() {
        return;
    }

    let engine = match ApiRelEngine::from_env() {
        Some(e) => e,
        None => return,
    };

    // Generic single-word entities that GLiNER sometimes extracts incorrectly
    let generic_words: std::collections::HashSet<&str> = [
        "services", "resources", "modules", "components", "systems",
        "applications", "tools", "frameworks", "libraries", "packages",
        "endpoints", "requests", "responses", "queries", "operations",
    ].into_iter().collect();

    // Only send candidates that likely need cleanup
    let needs_cleanup: Vec<usize> = entities
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            let word_count = e.text.split_whitespace().count();
            word_count > 1
                || generic_words.contains(e.text.to_lowercase().as_str())
        })
        .map(|(i, _)| i)
        .collect();

    if needs_cleanup.is_empty() {
        return;
    }

    let candidates: Vec<ExtractedEntity> = needs_cleanup
        .iter()
        .map(|&i| entities[i].clone())
        .collect();

    match engine.cleanup_entities(text, &candidates) {
        Ok(mapping) if !mapping.is_empty() => {
            // Remove entities the LLM dropped
            let indices_to_remove: Vec<usize> = needs_cleanup
                .iter()
                .filter(|&&i| !mapping.contains_key(&entities[i].text))
                .copied()
                .collect();

            // Apply renames
            for &i in &needs_cleanup {
                if let Some(cleaned) = mapping.get(&entities[i].text) {
                    if cleaned != &entities[i].text {
                        entities[i].text = cleaned.clone();
                    }
                }
            }

            // Remove dropped entities (reverse order to preserve indices)
            for &i in indices_to_remove.iter().rev() {
                entities.remove(i);
            }

            // Dedup after renaming
            let mut seen = std::collections::HashSet::new();
            entities.retain(|e| seen.insert(e.text.clone()));
        }
        Ok(_) => {}
        Err(_) => {}
    }
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
