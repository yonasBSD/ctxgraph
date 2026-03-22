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
const DEFAULT_MODEL: &str = "qwen2.5:7b";

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
        let prompt = build_prompt(text, entities, schema, &self.model);

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

        if self.model.contains("triplex") {
            parse_triplex_response(&body.response, entities, schema)
        } else {
            parse_llm_response(&body.response, entities, schema)
        }
    }
}

/// Build a Triplex-compatible prompt for relation extraction.
///
/// Triplex (SciPhi/Triplex) uses a specific prompt format and returns
/// `entities_and_triples` in `[id], name` / `[head_id] relation [tail_id]` format.
///
/// If the model is NOT Triplex (custom `CTXGRAPH_OLLAMA_MODEL`), uses a generic
/// JSON-lines prompt instead.
fn build_prompt(
    text: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
    model: &str,
) -> String {
    if model.contains("triplex") {
        build_triplex_prompt(text, entities, schema)
    } else {
        build_generic_prompt(text, entities, schema)
    }
}

/// Triplex-native prompt format.
fn build_triplex_prompt(
    text: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> String {
    let entity_list: String = entities
        .iter()
        .enumerate()
        .map(|(i, e)| format!("[{}], {}", i + 1, e.text))
        .collect::<Vec<_>>()
        .join(", ");

    let relation_keys = schema.relation_labels().join(", ");

    format!(
        r#"Perform the following task: Given the text and pre-identified entities, extract knowledge graph triplets that represent relationships between the entities.

Use ONLY these relation types: {relation_keys}

Pre-identified entities: {entity_list}

Text: {text}

Output the relationships as a JSON object with an "entities_and_triples" array. Entities are listed as "[id], name" and triples as "[head_id] relation [tail_id]". Only use the pre-identified entities and allowed relation types.

Output:"#,
    )
}

/// Generic few-shot prompt for non-Triplex models.
fn build_generic_prompt(
    text: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> String {
    let entity_list: Vec<String> = entities
        .iter()
        .map(|e| format!("- {} [{}]", e.text, e.entity_type))
        .collect();

    let relation_keys = schema.relation_labels().join(", ");

    format!(
        r#"You are a software architecture knowledge graph extractor. Given text and pre-identified entities, extract directed relationships as JSON lines.

### Relation types and direction rules

- chose: Person/Service/Component adopted a technology. head=chooser, tail=chosen.
  Example: "Alice chose PostgreSQL" → {{"head":"Alice","relation":"chose","tail":"PostgreSQL"}}
- rejected: Person/Service/Component rejected an alternative. head=rejector, tail=rejected.
  Example: "we decided against MongoDB" → {{"head":"Alice","relation":"rejected","tail":"MongoDB"}}
- replaced: NEW thing replaced OLD thing. head=NEW, tail=OLD. "from X to Y" means head=Y, tail=X.
  Example: "migrated from MySQL to PostgreSQL" → {{"head":"PostgreSQL","relation":"replaced","tail":"MySQL"}}
- depends_on: Consumer depends on provider. head=consumer, tail=provider.
  Example: "PaymentService uses Redis" → {{"head":"PaymentService","relation":"depends_on","tail":"Redis"}}
- fixed: Someone/something fixed an issue. head=fixer, tail=thing fixed.
  Example: "Bob patched the bug in AuthService" → {{"head":"Bob","relation":"fixed","tail":"AuthService"}}
- introduced: Introduced/added a new component. head=introducer, tail=introduced.
  Example: "BillingService added Prometheus monitoring" → {{"head":"BillingService","relation":"introduced","tail":"Prometheus"}}
- deprecated: Removed/phased out something. head=deprecator, tail=deprecated.
  Example: "Bob sunset the SOAP endpoint" → {{"head":"Bob","relation":"deprecated","tail":"SOAP"}}
- caused: Causal relationship, often to metrics. head=cause, tail=effect.
  Example: "Redis improved p99 latency" → {{"head":"Redis","relation":"caused","tail":"p99 latency"}}
- constrained_by: Something is constrained by a requirement. head=constrained, tail=constraint.
  Example: "Service must comply with the SLA" → {{"head":"Service","relation":"constrained_by","tail":"SLA"}}

### Critical direction rules (read twice)

1. "replaced": head = NEW, tail = OLD. "from X to Y" → head=Y, tail=X.
2. "depends_on": head = consumer, tail = provider.
3. "X over Y" in a choice context → X was chosen (chose), Y was rejected (rejected).

### Entities (use exact names)

{entity_list}

### Text

{text}

### Instructions

Output ONLY one JSON object per line. No markdown, no explanation, no extra text.
Each line: {{"head":"<entity>","relation":"<type>","tail":"<entity>"}}
- head and tail MUST be exact names from the entities list above.
- relation MUST be one of: {relation_keys}
- Only extract relationships explicitly supported by the text. Do not hallucinate.

Output:"#,
        entity_list = entity_list.join("\n"),
        relation_keys = relation_keys,
    )
}

/// Parse Triplex-format response into ExtractedRelation structs.
///
/// Triplex returns JSON like:
/// ```json
/// {"entities_and_triples": ["[1], Sarah", "[2], PostgreSQL", "[1] chose [2]"]}
/// ```
/// Entities are `[id], name` and triples are `[head_id] relation [tail_id]`.
fn parse_triplex_response(
    response: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> Result<Vec<ExtractedRelation>, OllamaError> {
    let relation_names: std::collections::HashSet<&str> =
        schema.relation_labels().into_iter().collect();

    // Find JSON in response (may have markdown code fences)
    let json_str = extract_json_block(response);

    #[derive(Deserialize)]
    struct TriplexOutput {
        entities_and_triples: Vec<String>,
    }

    let parsed: TriplexOutput = serde_json::from_str(json_str)
        .map_err(|e| OllamaError::Parse(format!("triplex JSON: {}", e)))?;

    // Phase 1: Build entity ID map from "[id], name" entries
    let mut id_to_name: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let entity_re = regex::Regex::new(r"^\[(\d+)\],?\s*(.+)$").unwrap();
    // Triple pattern: [head_id] relation text [tail_id]
    let triple_re = regex::Regex::new(r"^\[(\d+)\]\s+(.+?)\s+\[(\d+)\]$").unwrap();

    let mut triples_raw: Vec<(String, String, String)> = Vec::new();

    for item in &parsed.entities_and_triples {
        let item = item.trim();
        if let Some(caps) = entity_re.captures(item) {
            let id = caps[1].to_string();
            let name = caps[2].trim().to_string();
            id_to_name.insert(id, name);
        } else if let Some(caps) = triple_re.captures(item) {
            triples_raw.push((caps[1].to_string(), caps[2].trim().to_string(), caps[3].to_string()));
        }
    }

    // Phase 2: Resolve triples to entity names and validate
    let mut relations = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (head_id, rel_text, tail_id) in &triples_raw {
        let head_name = match id_to_name.get(head_id) {
            Some(n) => n,
            None => continue,
        };
        let tail_name = match id_to_name.get(tail_id) {
            Some(n) => n,
            None => continue,
        };

        // Match relation text to schema relation (case-insensitive, best match)
        let matched_rel = match_relation(&rel_text, &relation_names);
        let rel = match matched_rel {
            Some(r) => r,
            None => continue,
        };

        // Match entity names to actual extracted entities
        let head = match match_entity(head_name, entities) {
            Some(e) => e,
            None => continue,
        };
        let tail = match match_entity(tail_name, entities) {
            Some(e) => e,
            None => continue,
        };

        if head.text == tail.text {
            continue;
        }

        let key = (head.text.clone(), rel.to_string(), tail.text.clone());
        if seen.insert(key) {
            relations.push(ExtractedRelation {
                head: head.text.clone(),
                relation: rel.to_string(),
                tail: tail.text.clone(),
                confidence: 0.70,
            });
        }
    }

    Ok(relations)
}

/// Extract the first JSON object/array from a response that may contain markdown fences.
fn extract_json_block(response: &str) -> &str {
    // Try to find JSON inside ```json ... ``` blocks
    if let Some(start) = response.find("```json") {
        let after = &response[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    if let Some(start) = response.find("```") {
        let after = &response[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Try to find raw JSON object
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            return &response[start..=end];
        }
    }
    response.trim()
}

/// Match a Triplex relation string to a known schema relation.
///
/// Triplex may output free-form relation text like "chose", "depends on",
/// "replaced", etc. We match to the closest schema relation.
fn match_relation<'a>(text: &str, known: &std::collections::HashSet<&'a str>) -> Option<&'a str> {
    let text_lower = text.to_lowercase();
    let text_normalized = text_lower.replace(' ', "_");

    // Exact match (after normalization)
    if let Some(&r) = known.iter().find(|&&r| r == text_normalized) {
        return Some(r);
    }

    // Partial/stem match
    for &r in known {
        if text_normalized.contains(r) || r.contains(&*text_normalized) {
            return Some(r);
        }
        // Handle "depends on" → "depends_on"
        let r_spaced = r.replace('_', " ");
        if text_lower == r_spaced {
            return Some(r);
        }
    }

    None
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

// ---------------------------------------------------------------------------
// Entity cleanup via local LLM
// ---------------------------------------------------------------------------

/// Default model for entity cleanup (qwen2.5:3b tested best at 0.874 F1).
const DEFAULT_CLEANUP_MODEL: &str = "qwen2.5:3b";

/// Ollama-based entity name cleanup.
///
/// Takes GLiNER's raw entity extractions and uses a local LLM to:
/// - Shorten verbose names ("Terraform modules" → "Terraform")
/// - Remove generic/non-entity words ("services", "resources")
/// - Keep correct entities as-is
///
/// Benchmark results (on hard cases where GLiNER fails):
/// - GLiNER baseline: 0.636 F1
/// - qwen2.5:3b cleanup: 0.874 F1 (+37%)
pub struct OllamaEntityCleaner {
    base_url: String,
    model: String,
}

impl OllamaEntityCleaner {
    /// Create a new entity cleaner using env vars or defaults.
    ///
    /// Uses `CTXGRAPH_OLLAMA_URL` and `CTXGRAPH_CLEANUP_MODEL` env vars,
    /// falling back to localhost:11434 and qwen2.5:3b.
    pub fn new() -> Self {
        Self {
            base_url: std::env::var("CTXGRAPH_OLLAMA_URL")
                .unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string()),
            model: std::env::var("CTXGRAPH_CLEANUP_MODEL")
                .unwrap_or_else(|_| DEFAULT_CLEANUP_MODEL.to_string()),
        }
    }

    /// Check if Ollama is reachable.
    pub fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        reqwest::blocking::Client::new()
            .get(&url)
            .timeout(std::time::Duration::from_millis(500))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Clean up entity names using the local LLM.
    ///
    /// Returns a mapping from original entity text → cleaned entity text.
    /// Entities not in the result should be removed (LLM deemed them generic).
    pub fn cleanup(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> Result<std::collections::HashMap<String, String>, OllamaError> {
        if entities.is_empty() {
            return Ok(std::collections::HashMap::new());
        }

        let entity_names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        let entity_json = serde_json::to_string(&entity_names)
            .map_err(|e| OllamaError::Parse(e.to_string()))?;

        let prompt = format!(
            r#"A NER model extracted these entities from the text below. Some entity names may have extra words or be incorrect. Fix them.

Extracted entities: {entity_json}

Text: {text}

Rules:
- Fix entity names to be the shortest precise form (e.g., "Terraform modules" → "Terraform")
- Remove generic entities that aren't real names (e.g., "services", "resources")
- Keep entities that are correct as-is
- Return ONLY a JSON array of corrected entity name strings

Output ONLY a JSON array of strings, no other text."#
        );

        let request = OllamaRequest {
            model: self.model.clone(),
            prompt,
            stream: false,
            options: OllamaOptions {
                temperature: 0.0,
                num_predict: 256,
            },
        };

        let url = format!("{}/api/generate", self.base_url);
        let response = reqwest::blocking::Client::new()
            .post(&url)
            .timeout(std::time::Duration::from_secs(15))
            .json(&request)
            .send()
            .map_err(|e| OllamaError::Request(e.to_string()))?;

        if !response.status().is_success() {
            return Err(OllamaError::HttpStatus(response.status().as_u16()));
        }

        let body: OllamaResponse = response
            .json::<OllamaResponse>()
            .map_err(|e| OllamaError::Parse(e.to_string()))?;

        parse_cleanup_response(&body.response, &entity_names)
    }
}

/// Parse the LLM cleanup response into an original→cleaned name mapping.
///
/// The LLM returns a JSON array of cleaned entity name strings.
/// We match them back to originals by order and similarity.
fn parse_cleanup_response(
    response: &str,
    originals: &[&str],
) -> Result<std::collections::HashMap<String, String>, OllamaError> {
    let mut mapping = std::collections::HashMap::new();

    // Try to extract JSON array from response
    let text = response.trim();
    let json_str = if text.starts_with('[') {
        text
    } else {
        // Try markdown code block
        let re = regex::Regex::new(r"```(?:json)?\s*\n?(.*?)\n?```").unwrap();
        if let Some(caps) = re.captures(text) {
            caps.get(1).map(|m| m.as_str().trim()).unwrap_or(text)
        } else {
            text
        }
    };

    let cleaned: Vec<String> = serde_json::from_str(json_str)
        .map_err(|e| OllamaError::Parse(format!("cleanup JSON: {}", e)))?;

    // Build mapping: match cleaned names back to originals
    // Strategy: for each cleaned name, find the original it most likely came from
    let mut used_originals = std::collections::HashSet::new();

    for cleaned_name in &cleaned {
        let cleaned_lower = cleaned_name.to_lowercase();

        // Try exact match first
        if let Some(&orig) = originals.iter().find(|&&o| {
            o.to_lowercase() == cleaned_lower && !used_originals.contains(o)
        }) {
            mapping.insert(orig.to_string(), cleaned_name.clone());
            used_originals.insert(orig);
            continue;
        }

        // Try substring match (original contains cleaned or vice versa)
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

/// Cached Ollama cleanup availability check.
static OLLAMA_CLEANUP_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Apply Ollama entity cleanup to a set of extracted entities.
///
/// Conservative strategy: only modify entities that likely need fixing.
/// An entity is a candidate for cleanup if it:
/// - Contains multiple words (e.g., "Terraform modules", "IAM roles")
/// - Is a single generic word (e.g., "services", "resources")
///
/// Single-word proper nouns (e.g., "Kubernetes", "Redis") are NEVER sent
/// to the LLM — they're already correct and the LLM might remove them.
pub fn cleanup_entities(
    text: &str,
    entities: &mut Vec<ExtractedEntity>,
) {
    if std::env::var("CTXGRAPH_NO_OLLAMA").is_ok() {
        return;
    }

    let available = *OLLAMA_CLEANUP_AVAILABLE.get_or_init(|| {
        let cleaner = OllamaEntityCleaner::new();
        cleaner.is_available()
    });

    if !available {
        return;
    }

    // Generic single-word entities that GLiNER sometimes extracts incorrectly
    let generic_words: std::collections::HashSet<&str> = [
        "services", "resources", "modules", "components", "systems",
        "applications", "tools", "frameworks", "libraries", "packages",
        "endpoints", "requests", "responses", "queries", "operations",
    ].into_iter().collect();

    // Identify entities that need cleanup
    let needs_cleanup: Vec<usize> = entities
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            let word_count = e.text.split_whitespace().count();
            // Multi-word entities often have extra words
            word_count > 1
            // Single generic words should be removed
            || generic_words.contains(e.text.to_lowercase().as_str())
        })
        .map(|(i, _)| i)
        .collect();

    if needs_cleanup.is_empty() {
        return;
    }

    // Only send candidates to the LLM
    let candidates: Vec<ExtractedEntity> = needs_cleanup
        .iter()
        .map(|&i| entities[i].clone())
        .collect();

    let cleaner = OllamaEntityCleaner::new();
    match cleaner.cleanup(text, &candidates) {
        Ok(mapping) if !mapping.is_empty() => {
            // Remove generic single-word entities not in the mapping
            // and rename multi-word entities
            let indices_to_remove: Vec<usize> = needs_cleanup
                .iter()
                .filter(|&&i| !mapping.contains_key(&entities[i].text))
                .copied()
                .collect();

            // Apply renames first (before removing by index)
            for &i in &needs_cleanup {
                if let Some(cleaned) = mapping.get(&entities[i].text) {
                    if cleaned != &entities[i].text {
                        entities[i].text = cleaned.clone();
                    }
                }
            }

            // Remove entities the LLM dropped (iterate in reverse to preserve indices)
            for &i in indices_to_remove.iter().rev() {
                entities.remove(i);
            }

            // Dedup: after renaming, multiple entities might have the same text
            let mut seen = std::collections::HashSet::new();
            entities.retain(|e| seen.insert(e.text.clone()));
        }
        Ok(_) => {} // Empty mapping, no changes
        Err(_) => {} // Error, silently fall through
    }
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
