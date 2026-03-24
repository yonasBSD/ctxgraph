//! GLiREL ONNX backend — zero-shot relation extraction via split DeBERTa encoder + scoring head.
//!
//! Exported from `jackboyla/glirel-large-v0` as two ONNX models:
//!
//! 1. **encoder.onnx** — DeBERTa-v3-large (subword embeddings, fp16)
//!    - Input:  `input_ids` `[B, S]`, `attention_mask` `[B, S]`
//!    - Output: `last_hidden_state` `[B, S, 1024]` (fp16)
//!
//! 2. **scoring_head.onnx** — BiLSTM + span scorer
//!    - Input:  `word_rep` `[B, W, 768]`, `word_mask` `[B, W]`, `span_idx` `[B, E, 2]`,
//!              `relations_idx` `[B, P, 2, 2]`, `rel_type_rep` `[B, R, 768]`
//!    - Output: `relation_scores` `[B, P, R]`
//!
//! Intermediate steps (in Rust):
//! - Projection: Linear(1024 → 768) loaded from `.bin` weight files
//! - Scatter-mean: pool subword tokens → word-level representations
//! - Mean-pool: relation label encoder output → per-label vectors

use std::path::Path;

use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::ner::ExtractedEntity;
use crate::rel::{ExtractedRelation, RelError};

const ENCODER_DIM: usize = 1024;
const PROJ_DIM: usize = 768;

/// GLiREL zero-shot relation extraction engine (split ONNX).
pub struct GlirelEngine {
    encoder: Session,
    scoring_head: Session,
    tokenizer: Tokenizer,
    /// Projection weight [PROJ_DIM, ENCODER_DIM] stored row-major.
    proj_weight: Vec<f32>,
    /// Projection bias [PROJ_DIM].
    proj_bias: Vec<f32>,
}

impl GlirelEngine {
    /// Load split GLiREL model from a directory.
    ///
    /// Required files:
    /// - `encoder.onnx` (DeBERTa-v3-large)
    /// - `scoring_head.onnx` (BiLSTM + scorer)
    /// - `tokenizer.json` (DeBERTa tokenizer)
    /// - `projection_weight.bin` (f32, shape [768, 1024])
    /// - `projection_bias.bin` (f32, shape [768])
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

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let num_words = words.len();
        let num_entities = entities.len();
        let num_relations = relation_labels.len();

        // --- 1. Encode text → subword hidden states [S, 1024] ---
        let text_hidden = self.run_encoder(text)?;
        let seq_len = text_hidden.len() / ENCODER_DIM;

        // --- 2. Project 1024 → 768 ---
        let text_projected = self.project(&text_hidden, seq_len);

        // --- 3. Build words_mask and scatter-mean pool → word_rep [W, 768] ---
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| RelError::Inference(format!("tokenize text: {e}")))?;
        let words_mask = build_words_mask(&encoding, num_words);
        let word_rep = scatter_mean_pool(&text_projected, &words_mask, num_words, PROJ_DIM);

        // --- 4. Encode relation labels → rel_type_rep [R, 768] ---
        let rel_type_rep = self.encode_relation_labels(relation_labels)?;

        // --- 5. Build span_idx [E, 2] and relations_idx [P, 2, 2] ---
        let span_idx = entities_to_word_spans(text, &words, entities);
        let num_pairs = num_entities * (num_entities - 1);
        let relations_idx = build_relations_idx(&span_idx);

        // --- 6. Run scoring head ---
        let word_mask: Vec<i64> = vec![1i64; num_words];

        let t_word_rep = Tensor::from_array(([1, num_words, PROJ_DIM], word_rep))
            .map_err(|e| RelError::Inference(format!("tensor word_rep: {e}")))?;
        let t_word_mask = Tensor::from_array(([1, num_words], word_mask))
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

        // --- 7. Decode scores [1, P, R] ---
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

        // Sort by confidence descending, deduplicate per directed pair
        relations.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = std::collections::HashSet::new();
        relations.retain(|r| seen.insert((r.head_idx, r.tail_idx)));

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

    /// Run the DeBERTa encoder and return fp32 hidden states [S * ENCODER_DIM].
    fn run_encoder(&self, text: &str) -> Result<Vec<f32>, RelError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| RelError::Inference(format!("tokenize: {e}")))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let seq_len = input_ids.len();

        let t_ids = Tensor::from_array(([1, seq_len], input_ids))
            .map_err(|e| RelError::Inference(format!("tensor ids: {e}")))?;
        let t_mask = Tensor::from_array(([1, seq_len], attention_mask))
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

    /// Apply linear projection: x @ weight.T + bias  (per token).
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

    /// Encode relation labels: run encoder → project → mean pool per label.
    fn encode_relation_labels(&self, labels: &[&str]) -> Result<Vec<f32>, RelError> {
        let mut all_reps = Vec::with_capacity(labels.len() * PROJ_DIM);

        for &label in labels {
            let hidden = self.run_encoder(label)?;
            let seq_len = hidden.len() / ENCODER_DIM;
            let projected = self.project(&hidden, seq_len);

            // Mean pool (skip CLS/SEP by using tokens 1..seq_len-1, or all if short)
            let (start, end) = if seq_len > 2 { (1, seq_len - 1) } else { (0, seq_len) };
            let count = (end - start) as f32;

            for d in 0..PROJ_DIM {
                let sum: f32 = (start..end).map(|t| projected[t * PROJ_DIM + d]).sum();
                all_reps.push(sum / count);
            }
        }

        Ok(all_reps)
    }
}

/// Intermediate scored triple.
struct ScoredTriple {
    head_idx: usize,
    tail_idx: usize,
    relation: String,
    confidence: f32,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build words_mask: subword token index → word index (1-indexed, 0 = special).
fn build_words_mask(encoding: &tokenizers::Encoding, num_words: usize) -> Vec<i64> {
    let seq_len = encoding.get_ids().len();
    let mut mask = vec![0i64; seq_len];
    for (token_idx, word_id) in encoding.get_word_ids().iter().enumerate() {
        if let Some(wid) = word_id {
            if (*wid as usize) < num_words {
                mask[token_idx] = (*wid as i64) + 1; // 1-indexed
            }
        }
    }
    mask
}

/// Scatter-mean pool: subword tokens → word-level reps.
///
/// `words_mask[t]` = 0 means special token (skip), k>0 means word k (1-indexed).
fn scatter_mean_pool(
    token_reps: &[f32],
    words_mask: &[i64],
    num_words: usize,
    dim: usize,
) -> Vec<f32> {
    let mut sums = vec![0.0f32; num_words * dim];
    let mut counts = vec![0u32; num_words];

    for (t, &wid) in words_mask.iter().enumerate() {
        if wid <= 0 {
            continue;
        }
        let w = (wid - 1) as usize; // 0-indexed word
        if w >= num_words {
            continue;
        }
        counts[w] += 1;
        let src = t * dim;
        let dst = w * dim;
        for d in 0..dim {
            sums[dst + d] += token_reps[src + d];
        }
    }

    // Divide by count
    for w in 0..num_words {
        if counts[w] > 0 {
            let c = counts[w] as f32;
            let offset = w * dim;
            for d in 0..dim {
                sums[offset + d] /= c;
            }
        }
    }

    sums
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

/// Build relations_idx [num_pairs, 2, 2] for all directed entity pairs.
///
/// Each pair is `[[head_start, head_end], [tail_start, tail_end]]`.
fn build_relations_idx(span_idx: &[(i64, i64)]) -> Vec<i64> {
    let n = span_idx.len();
    let mut out = Vec::with_capacity(n * (n - 1) * 4);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                out.push(span_idx[i].0); // head start
                out.push(span_idx[i].1); // head end
                out.push(span_idx[j].0); // tail start
                out.push(span_idx[j].1); // tail end
            }
        }
    }
    out
}

/// Find an ONNX file from a list of candidates.
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

/// Load a raw f32 little-endian binary file.
fn load_f32_bin(path: &Path, expected_len: usize) -> Result<Vec<f32>, RelError> {
    let bytes = std::fs::read(path)
        .map_err(|e| RelError::ModelLoad(format!("read {}: {e}", path.display())))?;

    if bytes.len() != expected_len * 4 {
        return Err(RelError::ModelLoad(format!(
            "{}: expected {} f32 values ({} bytes), got {} bytes",
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
    fn test_scatter_mean_pool() {
        // 4 tokens, 2-dim, 2 words
        let reps = vec![
            0.0, 0.0, // token 0: special (CLS)
            1.0, 2.0, // token 1: word 0
            3.0, 4.0, // token 2: word 0 (subword continuation)
            5.0, 6.0, // token 3: word 1
        ];
        let mask = vec![0, 1, 1, 2]; // 0=special, 1=word0, 1=word0, 2=word1
        let pooled = scatter_mean_pool(&reps, &mask, 2, 2);
        // word 0: mean of (1,2) and (3,4) = (2, 3)
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 3.0).abs() < 1e-6);
        // word 1: just (5, 6)
        assert!((pooled[2] - 5.0).abs() < 1e-6);
        assert!((pooled[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_relations_idx() {
        let spans = vec![(0, 0), (3, 3), (5, 6)];
        let idx = build_relations_idx(&spans);
        // 3 entities → 6 pairs, 4 values each = 24
        assert_eq!(idx.len(), 24);
        // First pair: entity 0 → entity 1: [[0,0],[3,3]]
        assert_eq!(&idx[0..4], &[0, 0, 3, 3]);
        // Second pair: entity 0 → entity 2: [[0,0],[5,6]]
        assert_eq!(&idx[4..8], &[0, 0, 5, 6]);
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
        assert_eq!(spans[0], (0, 0)); // "Alice" = word 0
        assert_eq!(spans[1], (2, 2)); // "PostgreSQL" = word 2
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
        assert_eq!(spans[0], (2, 3)); // "Apache" = word 2, "Kafka" = word 3
    }
}
