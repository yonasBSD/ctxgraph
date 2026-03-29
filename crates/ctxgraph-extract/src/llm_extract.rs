//! LLM-based entity and relation extraction via OpenRouter API.
//!
//! Opt-in fallback for cross-domain extraction when local models (GLiNER/NuNER)
//! produce low-confidence results. Activated by setting `OPENROUTER_API_KEY`.
//!
//! Default model: `google/gemini-2.0-flash-001` (~$0.0003/episode).
//! Override with `CTXGRAPH_LLM_MODEL`.

use serde::{Deserialize, Serialize};

use crate::ner::ExtractedEntity;
use crate::rel::ExtractedRelation;
use crate::schema::ExtractionSchema;

const DEFAULT_MODEL: &str = "nvidia-auto";
const DEFAULT_URL: &str = "http://localhost:4000/v1/chat/completions";

/// LLM extraction engine — works with any OpenAI-compatible endpoint.
///
/// Supports: nvidia-litellm-router (free), OpenRouter, Ollama, OpenAI, Anthropic.
/// Configured via environment variables:
/// - `CTXGRAPH_LLM_URL`: API base URL (default: localhost:4000 for nvidia-litellm-router)
/// - `CTXGRAPH_LLM_KEY`: API key (or `OPENROUTER_API_KEY` for backward compat)
/// - `CTXGRAPH_LLM_MODEL`: Model name (default: nvidia-auto)
pub struct LlmExtractor {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
    url: String,
}

/// Combined extraction result from the LLM.
pub struct LlmExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
}

// --- JSON schema types for structured output ---

#[derive(Debug, Serialize, Deserialize)]
struct LlmResponse {
    entities: Vec<LlmEntity>,
    relations: Vec<LlmRelation>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LlmEntity {
    name: String,
    entity_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LlmRelation {
    head: String,
    relation: String,
    tail: String,
}

// --- OpenRouter API types ---

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    r#type: String,
    json_schema: JsonSchemaWrapper,
}

#[derive(Debug, Serialize)]
struct JsonSchemaWrapper {
    name: String,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {0}")]
    Api(String),

    #[error("parse error: {0}")]
    Parse(String),
}

impl LlmExtractor {
    /// Create a new LLM extractor from environment variables.
    ///
    /// Tries in order:
    /// 1. `CTXGRAPH_LLM_KEY` + `CTXGRAPH_LLM_URL` (explicit config)
    /// 2. `NVIDIA_API_KEY` (nvidia-litellm-router on localhost:4000)
    /// 3. `OPENROUTER_API_KEY` (OpenRouter cloud)
    ///
    /// Returns `None` if no API key is found.
    pub fn from_env() -> Option<Self> {
        let (api_key, default_url) = if let Ok(key) = std::env::var("CTXGRAPH_LLM_KEY") {
            if key.is_empty() {
                return None;
            }
            (key, DEFAULT_URL.to_string())
        } else if let Ok(key) = std::env::var("NVIDIA_API_KEY") {
            if key.is_empty() {
                return None;
            }
            // NVIDIA key present → assume nvidia-litellm-router on localhost:4000
            (
                "sk-litellm-master".to_string(),
                "http://localhost:4000/v1/chat/completions".to_string(),
            )
        } else if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
            if key.is_empty() {
                return None;
            }
            (
                key,
                "https://openrouter.ai/api/v1/chat/completions".to_string(),
            )
        } else {
            return None;
        };

        let url = std::env::var("CTXGRAPH_LLM_URL").unwrap_or(default_url);
        let model =
            std::env::var("CTXGRAPH_LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .ok()?;

        Some(Self {
            client,
            api_key,
            model,
            url,
        })
    }

    /// Extract entities only from text using the LLM.
    ///
    /// Used as fallback when GLiNER confidence is low. Relations are handled
    /// by GLiREL locally — the LLM only needs to find entity spans.
    pub fn extract_entities(
        &self,
        text: &str,
        schema: &ExtractionSchema,
    ) -> Result<Vec<ExtractedEntity>, LlmError> {
        let entity_types: Vec<String> = schema
            .entity_types
            .iter()
            .map(|(k, v)| format!("- {k}: {v}"))
            .collect();

        let system = format!(
            r#"Extract all named entities from the text. Return JSON only.

ENTITY TYPES:
{entity_list}

RULES:
- Use the SHORTEST canonical name for each entity. Examples:
  - "Cerner" not "Cerner EHR system"
  - "Blackboard" not "Blackboard LMS"
  - "FHIR R4" not "FHIR R4 APIs"
  - "Bloomberg Terminal" not "Bloomberg's API"
  - "blue-green deployment" not "zero-downtime blue-green deployment pattern"
- Strip articles (the, a, an) from the start of entity names.
- Entity names should be findable as substrings in the text.
- Each entity_type must be one of the types listed above.
- Extract ALL entities. Common things people miss:
  - Teams, departments, and roles count as Person (e.g., "radiology department", "compliance officer", "plant manager", "legal ops team")
  - Budget caps, SLAs, compliance requirements, certifications count as Constraint (e.g., "$50K budget cap", "HIPAA", "FedRAMP certification", "99.99% uptime SLA")
  - Incidents, errors, bottlenecks count as relevant entities (e.g., "ransomware incident", "T+3 settlement bottleneck")
  - Systems and platforms count as Component/Service even if not software (e.g., "trading engine", "clearing system", "predictive maintenance system")
- Do not invent entities not supported by the text."#,
            entity_list = entity_types.join("\n"),
        );

        let entity_type_enum: Vec<&str> = schema.entity_types.keys().map(|k| k.as_str()).collect();

        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "entity_type": {"type": "string", "enum": entity_type_enum}
                        },
                        "required": ["name", "entity_type"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": false
        });

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: system,
                },
                ChatMessage {
                    role: "user".into(),
                    content: format!("Extract entities from:\n\n{text}"),
                },
            ],
            response_format: Some(ResponseFormat {
                r#type: "json_schema".into(),
                json_schema: JsonSchemaWrapper {
                    name: "entity_result".into(),
                    strict: true,
                    schema: json_schema,
                },
            }),
            temperature: 0.0,
            max_tokens: 1024,
        };

        let response = self
            .client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(LlmError::Api(format!("{status}: {body}")));
        }

        let chat_resp: ChatResponse = response.json()?;
        let content = chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| LlmError::Parse("empty response".into()))?;

        let json_str = extract_json_from_response(content);

        #[derive(Deserialize)]
        struct EntityOnly {
            entities: Vec<LlmEntity>,
        }

        let parsed: EntityOnly = serde_json::from_str(json_str)
            .map_err(|e| LlmError::Parse(format!("JSON parse: {e}\nRaw: {content}")))?;

        Ok(parsed
            .entities
            .into_iter()
            .map(|e| {
                let name = canonicalize_llm_entity(&e.name, text);
                let span_start = text.to_lowercase().find(&name.to_lowercase()).unwrap_or(0);
                let span_end = if span_start > 0 {
                    span_start + name.len()
                } else {
                    // Fallback: try original name
                    text.find(&e.name).map(|s| s + e.name.len()).unwrap_or(0)
                };
                ExtractedEntity {
                    span_start,
                    span_end,
                    text: name,
                    entity_type: e.entity_type,
                    confidence: 1.0,
                }
            })
            .collect())
    }

    /// Extract entities and relations from text using the LLM (full extraction).
    ///
    /// Tries strict JSON schema first (works with large models like GPT-4o, Gemini).
    /// Falls back to prompt-based JSON (works with small models like Llama 3B, Qwen 3B).
    pub fn extract(
        &self,
        text: &str,
        schema: &ExtractionSchema,
    ) -> Result<LlmExtractionResult, LlmError> {
        // Try strict JSON schema first
        if let Ok(result) = self.extract_strict(text, schema) {
            return Ok(result);
        }
        // Fall back to prompt-based JSON for small models
        self.extract_prompt_json(text, schema)
    }

    fn extract_strict(
        &self,
        text: &str,
        schema: &ExtractionSchema,
    ) -> Result<LlmExtractionResult, LlmError> {
        let system_prompt = build_system_prompt(schema);
        let user_prompt = format!("Extract entities and relations from this text:\n\n{text}");

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".into(),
                    content: user_prompt,
                },
            ],
            response_format: Some(ResponseFormat {
                r#type: "json_schema".into(),
                json_schema: JsonSchemaWrapper {
                    name: "extraction_result".into(),
                    strict: true,
                    schema: extraction_json_schema(schema),
                },
            }),
            temperature: 0.0,
            max_tokens: 2048,
        };

        let response = self
            .client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(LlmError::Api(format!("{status}: {body}")));
        }

        let chat_resp: ChatResponse = response.json()?;
        let content = chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| LlmError::Parse("empty response".into()))?;

        let parsed: LlmResponse = serde_json::from_str(content)
            .map_err(|e| LlmError::Parse(format!("JSON parse: {e}\nRaw: {content}")))?;

        // Map to internal types
        let entities = parsed
            .entities
            .into_iter()
            .map(|e| ExtractedEntity {
                span_start: text.find(&e.name).unwrap_or(0),
                span_end: text.find(&e.name).map(|s| s + e.name.len()).unwrap_or(0),
                text: e.name,
                entity_type: e.entity_type,
                confidence: 1.0, // LLM doesn't provide confidence
            })
            .collect();

        let relations = parsed
            .relations
            .into_iter()
            .map(|r| ExtractedRelation {
                head: r.head,
                relation: r.relation,
                tail: r.tail,
                confidence: 1.0,
            })
            .collect();

        Ok(LlmExtractionResult {
            entities,
            relations,
        })
    }

    /// Fallback: prompt-based JSON extraction for small models that don't support
    /// structured output / json_schema response format.
    fn extract_prompt_json(
        &self,
        text: &str,
        schema: &ExtractionSchema,
    ) -> Result<LlmExtractionResult, LlmError> {
        let entity_types: Vec<String> = schema
            .entity_types
            .iter()
            .map(|(k, v)| format!("  - {k}: {v}"))
            .collect();
        let relation_types: Vec<String> = schema
            .relation_types
            .iter()
            .map(|(k, v)| format!("  - {k}: {}", v.description))
            .collect();

        let prompt = format!(
            r#"Extract entities and relations. ONLY output JSON, nothing else. /no_think

Entity types:
{entities}

Relation types:
{relations}

Rules:
- SHORT canonical names ("Redis" not "Redis server", "Stripe" not "Stripe SDK v2")
- Teams/departments/roles are Person
- Relation head and tail MUST exactly match an entity name you extracted
- For EVERY pair of related entities, add a relation

Text: "{text}"

{{"entities": [{{"name": "...", "entity_type": "..."}}], "relations": [{{"head": "exact entity name", "relation": "type", "tail": "exact entity name"}}]}}"#,
            entities = entity_types.join("\n"),
            relations = relation_types.join("\n"),
        );

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: prompt,
            }],
            response_format: None,
            temperature: 0.0,
            max_tokens: 2048,
        };

        let response = self
            .client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(LlmError::Api(format!("{status}: {body}")));
        }

        let chat_resp: ChatResponse = response.json()?;
        let raw = chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| LlmError::Parse("empty response".into()))?;

        // Extract JSON from the response — small models sometimes wrap it in markdown
        let json_str = extract_json_from_response(raw);

        let parsed: LlmResponse = serde_json::from_str(json_str)
            .map_err(|e| LlmError::Parse(format!("JSON parse: {e}\nRaw: {raw}")))?;

        let entities = parsed
            .entities
            .into_iter()
            .map(|e| {
                let name = canonicalize_llm_entity(&e.name, text);
                ExtractedEntity {
                    span_start: text.to_lowercase().find(&name.to_lowercase()).unwrap_or(0),
                    span_end: text
                        .to_lowercase()
                        .find(&name.to_lowercase())
                        .map(|s| s + name.len())
                        .unwrap_or(0),
                    text: name,
                    entity_type: e.entity_type,
                    confidence: 1.0,
                }
            })
            .collect();

        let relations = parsed
            .relations
            .into_iter()
            .map(|r| ExtractedRelation {
                head: r.head,
                relation: r.relation,
                tail: r.tail,
                confidence: 1.0,
            })
            .collect();

        Ok(LlmExtractionResult {
            entities,
            relations,
        })
    }
}

/// Extract JSON object from LLM response that may contain markdown fences,
/// `<think>` tags, or other preamble.
fn extract_json_from_response(raw: &str) -> &str {
    // Strip <think>...</think> tags (Qwen, DeepSeek reasoning models)
    let stripped = if let Some(think_end) = raw.find("</think>") {
        raw[think_end + 8..].trim()
    } else {
        raw
    };

    // Try to find JSON between ```json ... ``` markers
    if let Some(start) = stripped.find("```json") {
        let after = &stripped[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Try to find JSON between ``` ... ``` markers
    if let Some(start) = stripped.find("```") {
        let after = &stripped[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Try to find { ... } directly
    if let Some(start) = stripped.find('{')
        && let Some(end) = stripped.rfind('}')
    {
        return &stripped[start..=end];
    }
    stripped.trim()
}

/// Build the system prompt with schema-aware entity and relation types.
fn build_system_prompt(schema: &ExtractionSchema) -> String {
    let entity_types: Vec<String> = schema
        .entity_types
        .iter()
        .map(|(k, v)| format!("- {k}: {v}"))
        .collect();

    let relation_types: Vec<String> = schema
        .relation_types
        .iter()
        .map(|(k, v)| format!("- {k}: {}", v.description))
        .collect();

    format!(
        r#"You are an entity and relation extraction system.

ENTITY TYPES:
{entity_list}

RELATION TYPES:
{relation_list}

RULES:
- Use the SHORTEST canonical name for entities ("Redis" not "Redis cache", "Stripe" not "Stripe SDK v2").
- Teams, departments, and roles are Person entities ("platform team", "treasury department").
- Constraints include: compliance requirements, SLAs, certifications, budget caps.
- Each relation head and tail MUST be the EXACT name string from your entities list.
- CRITICAL: First extract all entities, then for EVERY pair of related entities, add a relation.
- Prefer specific relation types: "replaced" over "depends_on" when migration is described.
- Do not invent entities or relations not supported by the text."#,
        entity_list = entity_types.join("\n"),
        relation_list = relation_types.join("\n"),
    )
}

/// Build the JSON schema for structured output.
fn extraction_json_schema(schema: &ExtractionSchema) -> serde_json::Value {
    let entity_type_enum: Vec<&str> = schema.entity_types.keys().map(|k| k.as_str()).collect();
    let relation_enum: Vec<&str> = schema.relation_types.keys().map(|k| k.as_str()).collect();

    serde_json::json!({
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Exact entity name from text"},
                        "entity_type": {"type": "string", "enum": entity_type_enum}
                    },
                    "required": ["name", "entity_type"],
                    "additionalProperties": false
                }
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "head": {"type": "string", "description": "Head entity name"},
                        "relation": {"type": "string", "enum": relation_enum},
                        "tail": {"type": "string", "description": "Tail entity name"}
                    },
                    "required": ["head", "relation", "tail"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["entities", "relations"],
        "additionalProperties": false
    })
}

/// Canonicalize an LLM-extracted entity name to match what appears in text.
///
/// LLMs tend to return verbose names ("Cerner EHR", "Blackboard LMS",
/// "FHIR R4 APIs"). This strips common suffixes, articles, and possessives
/// to produce shorter canonical names that match text substrings.
fn canonicalize_llm_entity(name: &str, text: &str) -> String {
    let mut result = name.to_string();

    // Strip leading articles
    for prefix in ["the ", "The ", "a ", "A ", "an ", "An "] {
        if result.starts_with(prefix) {
            result = result[prefix.len()..].to_string();
        }
    }

    // Strip possessive 's (e.g., "Epic's MyChart" → "Epic MyChart")
    result = result.replace("'s ", " ");

    // Strip verbose suffixes ONLY when the trimmed version is a proper noun
    // or known entity. Avoid stripping meaningful parts like "report generation".
    let safe_suffixes = [
        " LMS",
        " EHR",
        " CRM",
        " APIs",
        " API",
        " system",
        " systems",
        " platform",
        " platforms",
    ];

    for suffix in &safe_suffixes {
        if result.ends_with(suffix) {
            let trimmed = &result[..result.len() - suffix.len()];
            // Only strip if trimmed is at least 3 chars and found in text
            if trimmed.len() >= 3 && text.to_lowercase().contains(&trimmed.to_lowercase()) {
                result = trimmed.to_string();
                break;
            }
        }
    }

    // If result doesn't appear in text but original does, keep original
    if !text.to_lowercase().contains(&result.to_lowercase())
        && text.to_lowercase().contains(&name.to_lowercase())
    {
        return name.to_string();
    }

    result
}
