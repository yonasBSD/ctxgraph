//! GLiREL ONNX backend — zero-shot relation extraction via split DeBERTa encoder + scoring head.
//!
//! GLiREL uses **joint prompt encoding**: relation labels are prepended to the text as:
//!   `[CLS] [REL] label1 [REL] label2 ... [SEP] word1 word2 ... [SEP]`
//! and the entire sequence is run through the encoder. Cross-attention between labels and
//! text tokens is what makes zero-shot scoring work.
//!
//! # Architecture
//!
//! 1. **encoder.onnx** — DeBERTa-v3-large (subword embeddings, fp16)
//! 2. **projection** — Linear(1024→768) loaded from `.bin` files
//! 3. **scoring_head.onnx** — BiLSTM + span scorer

use std::path::Path;

use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::ner::ExtractedEntity;
use crate::rel::{ExtractedRelation, RelError};

const ENCODER_DIM: usize = 1024;
const PROJ_DIM: usize = 768;
const CLS_ID: i64 = 1;
const SEP_ID: i64 = 2;
const REL_ID: i64 = 128002;

/// GLiREL zero-shot relation extraction engine (split ONNX).
pub struct GlirelEngine {
    encoder: Session,
    scoring_head: Session,
    tokenizer: Tokenizer,
    proj_weight: Vec<f32>, // [PROJ_DIM, ENCODER_DIM] row-major
    proj_bias: Vec<f32>,   // [PROJ_DIM]
}

impl GlirelEngine {
    /// Load split GLiREL model from a directory.
    pub fn new(model_dir: &Path) -> Result<Self, RelError> {
        let encoder_path = find_onnx(model_dir, &["encoder.onnx"])?;
        let scoring_path = find_onnx(model_dir, &["scoring_head.onnx"])?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weight_path = model_dir.join("projection_weight.bin");
        let bias_path = model_dir.join("projection_bias.bin");

        for (name, path) in [
            ("tokenizer", &tokenizer_path),
            ("projection_weight", &weight_path),
            ("projection_bias", &bias_path),
        ] {
            if !path.exists() {
                return Err(RelError::ModelLoad(format!(
                    "GLiREL {name} not found at {}",
                    path.display()
                )));
            }
        }

        let encoder = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.commit_from_file(&encoder_path))
            .map_err(|e| RelError::ModelLoad(format!("GLiREL encoder: {e}")))?;

        let scoring_head = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.commit_from_file(&scoring_path))
            .map_err(|e| RelError::ModelLoad(format!("GLiREL scoring_head: {e}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RelError::ModelLoad(format!("GLiREL tokenizer: {e}")))?;

        let proj_weight = load_f32_bin(&weight_path, PROJ_DIM * ENCODER_DIM)?;
        let proj_bias = load_f32_bin(&bias_path, PROJ_DIM)?;

        Ok(Self {
            encoder,
            scoring_head,
            tokenizer,
            proj_weight,
            proj_bias,
        })
    }

    /// Extract relations between pre-extracted entities using zero-shot scoring.
    pub fn extract(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        relation_labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<ExtractedRelation>, RelError> {
        if entities.len() < 2 || relation_labels.is_empty() || text.is_empty() {
            return Ok(Vec::new());
        }

        let text_words: Vec<&str> = text.split_whitespace().collect();
        if text_words.is_empty() {
            return Ok(Vec::new());
        }

        let num_entities = entities.len();
        let num_relations = relation_labels.len();

        // --- 1. Build prompt words: [REL] label1 [REL] label2 ... [SEP] ---
        let mut prompt_words: Vec<String> = Vec::new();
        for &label in relation_labels {
            prompt_words.push("[REL]".to_string());
            prompt_words.push(label.to_string());
        }
        prompt_words.push("[SEP]".to_string());
        let prompt_word_count = prompt_words.len(); // includes [SEP]

        // --- 2. Tokenize each word individually, concatenate with CLS/SEP ---
        let (input_ids, attention_mask, word_ids) =
            self.tokenize_words(&prompt_words, &text_words)?;
        let seq_len = input_ids.len();
        let num_total_words = prompt_word_count + text_words.len();

        // --- 3. Run encoder ---
        let hidden = self.run_encoder(&input_ids, &attention_mask, seq_len)?;

        // --- 4. Project 1024 → 768 ---
        let projected = self.project(&hidden, seq_len);

        // --- 5. First-subword pooling → word-level reps ---
        let word_rep = first_subword_pool(&projected, &word_ids, num_total_words, PROJ_DIM);

        // --- 6. Split: prompt labels (before [SEP]) and text words (after [SEP]) ---
        // Prompt words: 0=[REL], 1=label1, 2=[REL], 3=label2, ..., prompt_word_count-1=[SEP]
        // Label reps use "both" strategy: average each [REL]+label pair
        let _label_portion_len = prompt_word_count - 1; // exclude [SEP]
        let mut rel_type_rep = Vec::with_capacity(num_relations * PROJ_DIM);
        for r in 0..num_relations {
            let rel_word = r * 2; // [REL] token position
            let label_word = r * 2 + 1; // label token position
            let rel_offset = rel_word * PROJ_DIM;
            let label_offset = label_word * PROJ_DIM;
            for d in 0..PROJ_DIM {
                rel_type_rep.push((word_rep[rel_offset + d] + word_rep[label_offset + d]) / 2.0);
            }
        }

        // Text word reps start after prompt words
        let text_start = prompt_word_count * PROJ_DIM;
        let text_word_rep: Vec<f32> =
            word_rep[text_start..text_start + text_words.len() * PROJ_DIM].to_vec();
        let num_text_words = text_words.len();

        // --- 7. Build span_idx and relations_idx ---
        let span_idx = entities_to_word_spans(text, &text_words, entities);
        let num_pairs = num_entities * (num_entities - 1);
        let relations_idx = build_relations_idx(&span_idx);

        let word_mask: Vec<i64> = vec![1i64; num_text_words];

        // --- 8. Run scoring head ---
        let t_word_rep = Tensor::from_array(([1, num_text_words, PROJ_DIM], text_word_rep))
            .map_err(|e| RelError::Inference(format!("tensor word_rep: {e}")))?;
        let t_word_mask = Tensor::from_array(([1, num_text_words], word_mask))
            .map_err(|e| RelError::Inference(format!("tensor word_mask: {e}")))?;

        let span_flat: Vec<i64> = span_idx.iter().flat_map(|&(s, e)| [s, e]).collect();
        let t_span_idx = Tensor::from_array(([1, num_entities, 2], span_flat))
            .map_err(|e| RelError::Inference(format!("tensor span_idx: {e}")))?;

        let t_relations_idx = Tensor::from_array(([1, num_pairs, 2, 2], relations_idx))
            .map_err(|e| RelError::Inference(format!("tensor relations_idx: {e}")))?;

        let t_rel_type_rep = Tensor::from_array(([1, num_relations, PROJ_DIM], rel_type_rep))
            .map_err(|e| RelError::Inference(format!("tensor rel_type_rep: {e}")))?;

        let inputs = ort::inputs![
            "word_rep" => t_word_rep,
            "word_mask" => t_word_mask,
            "span_idx" => t_span_idx,
            "relations_idx" => t_relations_idx,
            "rel_type_rep" => t_rel_type_rep,
        ]
        .map_err(|e| RelError::Inference(format!("scoring inputs: {e}")))?;

        let outputs = self
            .scoring_head
            .run(inputs)
            .map_err(|e| RelError::Inference(format!("scoring_head run: {e}")))?;

        // --- 9. Decode scores [1, P, R] ---
        let scores_value = outputs
            .get("relation_scores")
            .ok_or_else(|| RelError::Inference("missing relation_scores".into()))?;
        let scores_view = scores_value
            .try_extract_tensor::<f32>()
            .map_err(|e| RelError::Inference(format!("extract scores: {e}")))?;
        let scores = scores_view
            .as_slice()
            .ok_or_else(|| RelError::Inference("non-contiguous scores".into()))?;

        // Map pair index → (head_entity_idx, tail_entity_idx)
        let mut pair_map = Vec::with_capacity(num_pairs);
        for i in 0..num_entities {
            for j in 0..num_entities {
                if i != j {
                    pair_map.push((i, j));
                }
            }
        }

        let mut relations = Vec::new();
        for (pair_idx, &(head_idx, tail_idx)) in pair_map.iter().enumerate() {
            for (rel_idx, &rel_label) in relation_labels.iter().enumerate() {
                let flat_idx = pair_idx * num_relations + rel_idx;
                if flat_idx >= scores.len() {
                    break;
                }
                let conf = sigmoid(scores[flat_idx]);
                if conf >= threshold {
                    relations.push(ScoredTriple {
                        head_idx,
                        tail_idx,
                        relation: rel_label.to_string(),
                        confidence: conf,
                    });
                }
            }
        }

        // Sort by confidence descending
        relations.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate: for each undirected entity pair + relation type, keep the
        // direction with higher confidence. This fixes the "both directions score
        // similarly" issue — we pick the best one.
        let mut seen_undirected = std::collections::HashSet::new();
        relations.retain(|r| {
            let key = if r.head_idx < r.tail_idx {
                (r.head_idx, r.tail_idx, r.relation.clone())
            } else {
                (r.tail_idx, r.head_idx, r.relation.clone())
            };
            seen_undirected.insert(key)
        });

        // Text-context boost: if the relation label (or a close synonym) appears
        // in the text near the entity pair, boost its confidence. This helps GLiREL
        // pick "replaced" over "chose" when the text literally says "replaced".
        let text_lower = text.to_lowercase();
        let context_keywords: &[(&str, &[&str])] = &[
            (
                "chose",
                &[
                    "chose",
                    "choose",
                    "selected",
                    "picked",
                    "adopted",
                    "decided to use",
                    "went with",
                ],
            ),
            (
                "rejected",
                &["rejected", "decided against", "ruled out", "passed on"],
            ),
            (
                "replaced",
                &[
                    "replaced",
                    "migrated from",
                    "switched from",
                    "moved from",
                    "transitioned",
                ],
            ),
            (
                "depends_on",
                &["depends on", "relies on", "built on", "running on", "uses"],
            ),
            (
                "introduced",
                &[
                    "introduced",
                    "added",
                    "implemented",
                    "rolled out",
                    "adopted",
                ],
            ),
            (
                "deprecated",
                &["deprecated", "removed", "phased out", "sunset", "dropped"],
            ),
            ("caused", &["caused", "resulted in", "led to", "triggered"]),
            ("fixed", &["fixed", "resolved", "patched", "addressed"]),
            (
                "constrained_by",
                &[
                    "constrained by",
                    "limited by",
                    "requirement",
                    "compliance",
                    "must",
                ],
            ),
        ];
        for r in &mut relations {
            for &(rel_name, keywords) in context_keywords {
                if r.relation == rel_name {
                    if keywords.iter().any(|kw| text_lower.contains(kw)) {
                        r.confidence *= 1.3; // 30% boost if keyword found in text
                    }
                    break;
                }
            }
        }

        // Re-sort after boost
        relations.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate per directed pair (keep highest-confidence relation type)
        let mut seen_directed = std::collections::HashSet::new();
        relations.retain(|r| seen_directed.insert((r.head_idx, r.tail_idx)));

        Ok(relations
            .into_iter()
            .map(|r| ExtractedRelation {
                head: entities[r.head_idx].text.clone(),
                relation: r.relation,
                tail: entities[r.tail_idx].text.clone(),
                confidence: r.confidence as f64,
            })
            .collect())
    }

    /// Tokenize prompt words and text words individually, concatenating into a single sequence.
    ///
    /// Returns `(input_ids, attention_mask, word_ids)` where `word_ids[t]` maps subword
    /// token `t` to its word index (or `None` for CLS/SEP).
    #[allow(clippy::type_complexity)]
    fn tokenize_words(
        &self,
        prompt_words: &[String],
        text_words: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<Option<usize>>), RelError> {
        let mut input_ids = vec![CLS_ID];
        let mut word_ids: Vec<Option<usize>> = vec![None]; // CLS

        let mut word_idx = 0usize;
        for word in prompt_words {
            if word == "[REL]" {
                input_ids.push(REL_ID);
                word_ids.push(Some(word_idx));
            } else if word == "[SEP]" {
                input_ids.push(SEP_ID);
                word_ids.push(Some(word_idx));
            } else {
                let enc = self
                    .tokenizer
                    .encode(word.as_str(), false)
                    .map_err(|e| RelError::Inference(format!("tokenize '{word}': {e}")))?;
                for &id in enc.get_ids() {
                    input_ids.push(id as i64);
                    word_ids.push(Some(word_idx));
                }
            }
            word_idx += 1;
        }

        for &word in text_words {
            let enc = self
                .tokenizer
                .encode(word, false)
                .map_err(|e| RelError::Inference(format!("tokenize '{word}': {e}")))?;
            for &id in enc.get_ids() {
                input_ids.push(id as i64);
                word_ids.push(Some(word_idx));
            }
            word_idx += 1;
        }

        input_ids.push(SEP_ID); // trailing [SEP]
        word_ids.push(None);

        let attention_mask = vec![1i64; input_ids.len()];
        Ok((input_ids, attention_mask, word_ids))
    }

    /// Run the DeBERTa encoder and return fp32 hidden states [seq_len * ENCODER_DIM].
    fn run_encoder(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        seq_len: usize,
    ) -> Result<Vec<f32>, RelError> {
        let t_ids = Tensor::from_array(([1, seq_len], input_ids.to_vec()))
            .map_err(|e| RelError::Inference(format!("tensor ids: {e}")))?;
        let t_mask = Tensor::from_array(([1, seq_len], attention_mask.to_vec()))
            .map_err(|e| RelError::Inference(format!("tensor mask: {e}")))?;

        let inputs = ort::inputs![
            "input_ids" => t_ids,
            "attention_mask" => t_mask,
        ]
        .map_err(|e| RelError::Inference(format!("encoder inputs: {e}")))?;

        let outputs = self
            .encoder
            .run(inputs)
            .map_err(|e| RelError::Inference(format!("encoder run: {e}")))?;

        let hidden_value = outputs
            .get("last_hidden_state")
            .ok_or_else(|| RelError::Inference("missing last_hidden_state".into()))?;

        // Try fp32 first, then fp16
        if let Ok(view) = hidden_value.try_extract_tensor::<f32>() {
            Ok(view
                .as_slice()
                .ok_or_else(|| RelError::Inference("non-contiguous hidden".into()))?
                .to_vec())
        } else {
            let view = hidden_value
                .try_extract_tensor::<f16>()
                .map_err(|e| RelError::Inference(format!("extract hidden fp16: {e}")))?;
            let slice = view
                .as_slice()
                .ok_or_else(|| RelError::Inference("non-contiguous hidden fp16".into()))?;
            Ok(slice.iter().map(|h| h.to_f32()).collect())
        }
    }

    /// Apply linear projection: x @ weight.T + bias (per token).
    fn project(&self, hidden: &[f32], num_tokens: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; num_tokens * PROJ_DIM];
        for t in 0..num_tokens {
            let h_offset = t * ENCODER_DIM;
            let o_offset = t * PROJ_DIM;
            for o in 0..PROJ_DIM {
                let mut sum = self.proj_bias[o];
                let w_offset = o * ENCODER_DIM;
                for i in 0..ENCODER_DIM {
                    sum += hidden[h_offset + i] * self.proj_weight[w_offset + i];
                }
                out[o_offset + o] = sum;
            }
        }
        out
    }

    /// Extract relations with schema-aware direction resolution.
    ///
    /// Uses head/tail entity type constraints from the schema to fix direction:
    /// if GLiREL says `Database→chose→Person` but the schema says `chose` has
    /// `head=["Person"]` and `tail=["Database"]`, it flips to `Person→chose→Database`.
    pub fn extract_with_schema(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        schema: &crate::schema::ExtractionSchema,
        threshold: f32,
    ) -> Result<Vec<ExtractedRelation>, RelError> {
        let relation_labels: Vec<&str> = schema.relation_labels();
        let mut relations = self.extract(text, entities, &relation_labels, threshold)?;

        for rel in &mut relations {
            if let Some(spec) = schema.relation_types.get(&rel.relation) {
                let head_ent = entities.iter().find(|e| e.text == rel.head);
                let tail_ent = entities.iter().find(|e| e.text == rel.tail);

                if let (Some(h), Some(t)) = (head_ent, tail_ent) {
                    let head_valid = spec.head.iter().any(|ht| ht == &h.entity_type);
                    let tail_valid = spec.tail.iter().any(|tt| tt == &t.entity_type);
                    let flip_head_valid = spec.head.iter().any(|ht| ht == &t.entity_type);
                    let flip_tail_valid = spec.tail.iter().any(|tt| tt == &h.entity_type);

                    if !head_valid || !tail_valid {
                        // Current direction invalid — flip if the reverse works
                        if flip_head_valid && flip_tail_valid {
                            std::mem::swap(&mut rel.head, &mut rel.tail);
                        }
                    } else if flip_head_valid && flip_tail_valid {
                        // Both directions are schema-valid — prefer the one where
                        // head appears before tail in text (natural reading order).
                        // E.g., "Alice chose PostgreSQL" → Alice is head (appears first).
                        if h.span_start > t.span_start {
                            std::mem::swap(&mut rel.head, &mut rel.tail);
                        }
                    }
                }
            }
        }

        Ok(relations)
    }
}

struct ScoredTriple {
    head_idx: usize,
    tail_idx: usize,
    relation: String,
    confidence: f32,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// First-subword pooling: take the first subword token's representation for each word.
fn first_subword_pool(
    token_reps: &[f32],
    word_ids: &[Option<usize>],
    num_words: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_words * dim];
    let mut seen = vec![false; num_words];

    for (t, wid_opt) in word_ids.iter().enumerate() {
        if let Some(w) = *wid_opt
            && w < num_words
            && !seen[w]
        {
            seen[w] = true;
            let src = t * dim;
            let dst = w * dim;
            out[dst..dst + dim].copy_from_slice(&token_reps[src..src + dim]);
        }
    }

    out
}

/// Map entity char offsets → word-level (start, end) indices.
fn entities_to_word_spans(
    text: &str,
    words: &[&str],
    entities: &[ExtractedEntity],
) -> Vec<(i64, i64)> {
    let mut word_char_spans = Vec::with_capacity(words.len());
    let mut byte_pos = 0;

    for &word in words {
        if let Some(rel_pos) = text[byte_pos..].find(word) {
            let abs_byte = byte_pos + rel_pos;
            let char_start = text[..abs_byte].chars().count();
            let char_end = char_start + word.chars().count();
            word_char_spans.push((char_start, char_end));
            byte_pos = abs_byte + word.len();
        } else {
            let prev_end = word_char_spans.last().map(|&(_, e)| e).unwrap_or(0);
            word_char_spans.push((prev_end, prev_end + word.chars().count()));
        }
    }

    entities
        .iter()
        .map(|ent| {
            let mut start_word = 0i64;
            let mut end_word = 0i64;
            let mut found = false;

            for (word_idx, &(ws, we)) in word_char_spans.iter().enumerate() {
                if we > ent.span_start && ws < ent.span_end {
                    if !found {
                        start_word = word_idx as i64;
                        found = true;
                    }
                    end_word = word_idx as i64;
                }
            }

            (start_word, end_word)
        })
        .collect()
}

/// Build relations_idx [num_pairs * 4] for all directed entity pairs.
fn build_relations_idx(span_idx: &[(i64, i64)]) -> Vec<i64> {
    let n = span_idx.len();
    let mut out = Vec::with_capacity(n * (n - 1) * 4);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                out.push(span_idx[i].0);
                out.push(span_idx[i].1);
                out.push(span_idx[j].0);
                out.push(span_idx[j].1);
            }
        }
    }
    out
}

fn find_onnx(dir: &Path, candidates: &[&str]) -> Result<std::path::PathBuf, RelError> {
    for name in candidates {
        let p = dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(RelError::ModelLoad(format!(
        "ONNX not found in {}: tried {:?}",
        dir.display(),
        candidates
    )))
}

fn load_f32_bin(path: &Path, expected_len: usize) -> Result<Vec<f32>, RelError> {
    let bytes = std::fs::read(path)
        .map_err(|e| RelError::ModelLoad(format!("read {}: {e}", path.display())))?;
    if bytes.len() != expected_len * 4 {
        return Err(RelError::ModelLoad(format!(
            "{}: expected {} f32 ({} bytes), got {} bytes",
            path.display(),
            expected_len,
            expected_len * 4,
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_first_subword_pool() {
        // 5 tokens, 2-dim, 2 words. Word 0 has 2 subwords, word 1 has 1.
        let reps = vec![
            0.0, 0.0, // token 0: CLS (no word)
            1.0, 2.0, // token 1: word 0, first subword
            3.0, 4.0, // token 2: word 0, second subword
            5.0, 6.0, // token 3: word 1
            0.0, 0.0, // token 4: SEP (no word)
        ];
        let wids = vec![None, Some(0), Some(0), Some(1), None];
        let pooled = first_subword_pool(&reps, &wids, 2, 2);
        // word 0: first subword = (1, 2)
        assert!((pooled[0] - 1.0).abs() < 1e-6);
        assert!((pooled[1] - 2.0).abs() < 1e-6);
        // word 1: (5, 6)
        assert!((pooled[2] - 5.0).abs() < 1e-6);
        assert!((pooled[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_relations_idx() {
        let spans = vec![(0, 0), (3, 3), (5, 6)];
        let idx = build_relations_idx(&spans);
        assert_eq!(idx.len(), 24); // 3*2 pairs * 4 values
        assert_eq!(&idx[0..4], &[0, 0, 3, 3]); // entity 0 → entity 1
        assert_eq!(&idx[4..8], &[0, 0, 5, 6]); // entity 0 → entity 2
    }

    #[test]
    fn test_entities_to_word_spans() {
        let text = "Alice chose PostgreSQL over MySQL";
        let words: Vec<&str> = text.split_whitespace().collect();
        let entities = vec![
            ExtractedEntity {
                text: "Alice".into(),
                entity_type: "Person".into(),
                span_start: 0,
                span_end: 5,
                confidence: 0.9,
            },
            ExtractedEntity {
                text: "PostgreSQL".into(),
                entity_type: "Database".into(),
                span_start: 12,
                span_end: 22,
                confidence: 0.9,
            },
        ];
        let spans = entities_to_word_spans(text, &words, &entities);
        assert_eq!(spans[0], (0, 0));
        assert_eq!(spans[1], (2, 2));
    }

    #[test]
    fn test_entities_to_word_spans_multiword() {
        let text = "We chose Apache Kafka for messaging";
        let words: Vec<&str> = text.split_whitespace().collect();
        let entities = vec![ExtractedEntity {
            text: "Apache Kafka".into(),
            entity_type: "Component".into(),
            span_start: 9,
            span_end: 21,
            confidence: 0.9,
        }];
        let spans = entities_to_word_spans(text, &words, &entities);
        assert_eq!(spans[0], (2, 3));
    }
}
