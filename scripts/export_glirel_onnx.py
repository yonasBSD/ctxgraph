#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "glirel>=1.0.0",
#     "torch>=2.0.0",
#     "onnx>=1.14.0",
#     "onnxruntime>=1.16.0",
#     "transformers>=4.30.0",
#     "optimum>=1.16.0",
#     "numpy>=1.21.0",
# ]
# ///
"""
Export GLiREL (relation extraction) model to ONNX format for use in ctxgraph.

Exports as two ONNX models to avoid OOM during tracing of the large encoder:

1. encoder.onnx - DeBERTa-v3-large encoder (exported via optimum, memory-efficient)
   Inputs:  input_ids [B, S], attention_mask [B, S]
   Outputs: last_hidden_state [B, S, 1024]

2. scoring_head.onnx - Post-encoder scoring pipeline (small, easy to trace)
   Inputs:
     word_rep          [B, max_words, 1024]
     word_mask         [B, max_words]
     span_idx          [B, num_entities, 2]
     relations_idx     [B, num_pairs, 2, 2]
     rel_type_rep      [B, num_relations, 1024]
   Outputs:
     relation_scores   [B, num_pairs, num_relations]

Inference pipeline (Rust/Python):
  1. Tokenize text, build word_ids mapping
  2. Run encoder.onnx on text tokens -> hidden_states
  3. scatter_mean pool hidden_states by word_ids -> word_rep
  4. Run encoder.onnx on relation label tokens -> label_hidden
  5. Mean-pool label_hidden -> raw_label_rep
  6. Project + FFN on raw_label_rep (done in scoring_head.onnx or pre-computed)
  7. Run scoring_head.onnx(word_rep, word_mask, span_idx, relations_idx, rel_type_rep)

Usage:
    python scripts/export_glirel_onnx.py
    python scripts/export_glirel_onnx.py --model jackboyla/glirel-large-v0
    python scripts/export_glirel_onnx.py --output ~/.cache/ctxgraph/models/glirel-large-v0/
    python scripts/export_glirel_onnx.py --quantize
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEFAULT_MODEL = "jackboyla/glirel-large-v0"
DEFAULT_OUTPUT = os.path.expanduser("~/.cache/ctxgraph/models/glirel-large-v0")


class ScoringHead(nn.Module):
    """
    Post-encoder scoring pipeline for GLiREL.

    Takes pre-computed word-level and relation label representations,
    runs BiLSTM + span representation + scorer.
    """

    def __init__(self, glirel_model):
        super().__init__()

        # BiLSTM (raw LSTM, no pack_padded_sequence)
        self.lstm = glirel_model.rnn.lstm

        # Span and relation representation
        self.span_rep_layer = glirel_model.span_rep_layer

        # Prompt representation FFN (applied to relation label embeddings)
        self.prompt_rep_layer = glirel_model.prompt_rep_layer

        # Scorer
        self.scorer = glirel_model.scorer

        # Refinement layers
        self.has_refine_relation = hasattr(glirel_model, "refine_relation")
        self.has_refine_prompt = hasattr(glirel_model, "refine_prompt")
        if self.has_refine_relation:
            self.refine_relation = glirel_model.refine_relation
        if self.has_refine_prompt:
            self.refine_prompt = glirel_model.refine_prompt

    def forward(
        self,
        word_rep: torch.Tensor,
        word_mask: torch.Tensor,
        span_idx: torch.Tensor,
        relations_idx: torch.Tensor,
        rel_type_rep: torch.Tensor,
    ) -> torch.Tensor:
        B = word_rep.shape[0]

        # BiLSTM
        lstm_out, _ = self.lstm(word_rep)
        lstm_out = lstm_out * word_mask.unsqueeze(-1).float()

        # Entity pair representations
        rel_rep = self.span_rep_layer(lstm_out, span_idx=span_idx, relations_idx=relations_idx)

        # Apply prompt FFN to relation labels
        rel_type_rep = self.prompt_rep_layer(rel_type_rep)

        # Optional refinement
        if self.has_refine_relation:
            num_rel = rel_type_rep.shape[1]
            rel_type_mask = torch.ones(B, num_rel, dtype=torch.bool, device=word_rep.device)
            rel_rep_mask = torch.ones(B, rel_rep.shape[1], dtype=torch.bool, device=word_rep.device)
            rel_rep = self.refine_relation(rel_rep, lstm_out, rel_rep_mask, word_mask)
        if self.has_refine_prompt:
            if not self.has_refine_relation:
                num_rel = rel_type_rep.shape[1]
                rel_type_mask = torch.ones(B, num_rel, dtype=torch.bool, device=word_rep.device)
                rel_rep_mask = torch.ones(B, rel_rep.shape[1], dtype=torch.bool, device=word_rep.device)
            rel_type_rep = self.refine_prompt(rel_type_rep, rel_rep, rel_type_mask, rel_rep_mask)

        scores = self.scorer(rel_rep, rel_type_rep)
        return scores


def export_encoder_optimum(model, tokenizer, out_dir: Path, opset: int):
    """
    Export the DeBERTa encoder using HuggingFace optimum.
    This is memory-efficient and handles the complex DeBERTa architecture.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoConfig

    # Save the encoder temporarily as a standalone HF model
    tmp_dir = out_dir / "_tmp_encoder"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    encoder = model.token_rep_layer.bert_layer.model
    encoder.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))

    # Also save config
    encoder.config.save_pretrained(str(tmp_dir))

    print("[glirel-export] Exporting encoder via optimum...")

    # Use optimum to export
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        str(tmp_dir),
        export=True,
    )

    # Save the ONNX model
    ort_model.save_pretrained(str(out_dir / "_tmp_onnx"))

    # Move the model file
    onnx_src = out_dir / "_tmp_onnx" / "model.onnx"
    onnx_dst = out_dir / "encoder.onnx"
    if onnx_src.exists():
        shutil.move(str(onnx_src), str(onnx_dst))
    else:
        # Check for external data format
        for f in (out_dir / "_tmp_onnx").iterdir():
            if f.suffix == ".onnx":
                shutil.move(str(f), str(out_dir / f.name.replace("model", "encoder")))
            elif f.suffix == ".onnx_data":
                shutil.move(str(f), str(out_dir / f.name.replace("model", "encoder")))

    # Cleanup temp dirs
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    shutil.rmtree(str(out_dir / "_tmp_onnx"), ignore_errors=True)

    enc_path = out_dir / "encoder.onnx"
    if enc_path.exists():
        size_mb = enc_path.stat().st_size / 1024 / 1024
        print(f"[glirel-export] Exported encoder.onnx ({size_mb:.1f} MB)")
        return enc_path
    else:
        print("[glirel-export] ERROR: encoder.onnx not found after export")
        return None


def export_scoring_head(model, hidden_size: int, out_dir: Path, opset: int):
    """Export the scoring head using legacy TorchScript ONNX exporter."""
    print("[glirel-export] Exporting scoring head...")

    scoring = ScoringHead(model)
    scoring.eval()
    scoring = scoring.to("cpu")

    # Create dummy inputs
    B, max_words, D = 1, 7, hidden_size
    word_rep = torch.randn(B, max_words, D)
    word_mask = torch.ones(B, max_words, dtype=torch.long)
    span_idx = torch.tensor([[[0, 0], [3, 3], [5, 5]]], dtype=torch.long)
    relations_idx = torch.tensor([
        [[[0, 0], [3, 3]], [[3, 3], [0, 0]],
         [[0, 0], [5, 5]], [[5, 5], [0, 0]],
         [[3, 3], [5, 5]], [[5, 5], [3, 3]]],
    ], dtype=torch.long)
    rel_type_rep = torch.randn(B, 2, D)

    dummy = (word_rep, word_mask, span_idx, relations_idx, rel_type_rep)

    # Verify
    with torch.no_grad():
        out = scoring(*dummy)
    print(f"  Scoring head output: {out.shape}, range [{out.min():.4f}, {out.max():.4f}]")

    score_path = out_dir / "scoring_head.onnx"
    torch.onnx.export(
        scoring,
        dummy,
        str(score_path),
        input_names=["word_rep", "word_mask", "span_idx", "relations_idx", "rel_type_rep"],
        output_names=["relation_scores"],
        dynamic_axes={
            "word_rep": {0: "batch", 1: "max_words"},
            "word_mask": {0: "batch", 1: "max_words"},
            "span_idx": {0: "batch", 1: "num_entities"},
            "relations_idx": {0: "batch", 1: "num_pairs"},
            "rel_type_rep": {0: "batch", 1: "num_relations"},
            "relation_scores": {0: "batch", 1: "num_pairs", 2: "num_relations"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = score_path.stat().st_size / 1024 / 1024
    print(f"[glirel-export] Exported scoring_head.onnx ({size_mb:.1f} MB)")

    return score_path, out.detach().numpy(), dummy


def export_glirel(
    model_name: str,
    output_dir: str,
    quantize: bool = False,
    opset: int = 14,
) -> None:
    """Export GLiREL model to ONNX (split encoder + scoring head)."""
    from glirel import GLiREL

    print(f"[glirel-export] Loading model: {model_name}")
    model = GLiREL.from_pretrained(model_name)
    model.eval()
    model = model.to("cpu")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer = model.token_rep_layer.bert_layer.tokenizer
    tokenizer.save_pretrained(str(out))
    print(f"[glirel-export] Saved tokenizer to {out}")

    # Save model config
    config_dict = vars(model.config) if hasattr(model.config, "__dict__") else {}
    hidden_size = config_dict.get("hidden_size", 1024)
    config_export = {
        "model_name": model_name,
        "architecture": "glirel",
        "hidden_size": hidden_size,
        "max_width": config_dict.get("max_width", 12),
        "subtoken_pooling": config_dict.get("subtoken_pooling", "first"),
        "span_marker_mode": config_dict.get("span_marker_mode", "markerv1"),
        "scorer": config_dict.get("scorer", "dot"),
        "label_embed_strategy": config_dict.get("label_embed_strategy", "ent_token"),
        "rel_mode": config_dict.get("rel_mode", "marker"),
        "has_projection": hasattr(model.token_rep_layer, "projection"),
        "split_export": True,
        "encoder_file": "encoder.onnx",
        "scoring_head_file": "scoring_head.onnx",
    }

    with open(out / "glirel_config.json", "w") as f:
        json.dump(config_export, f, indent=2)
    print(f"[glirel-export] Saved config")

    # === Part 1: Export encoder via optimum ===
    enc_path = export_encoder_optimum(model, tokenizer, out, opset)

    # Free encoder from original model to save memory
    gc.collect()

    # === Part 2: Export scoring head ===
    score_path, pt_scores, score_dummy = export_scoring_head(model, hidden_size, out, opset)

    # Free model
    del model
    gc.collect()

    # === Part 3: INT8 Quantization ===
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for name, path in [("encoder", enc_path), ("scoring_head", score_path)]:
            if path is not None and path.exists():
                quant_path = out / (path.stem + "_int8.onnx")
                print(f"[glirel-export] Quantizing {name}...")
                try:
                    quantize_dynamic(
                        str(path),
                        str(quant_path),
                        weight_type=QuantType.QInt8,
                    )
                    orig_mb = path.stat().st_size / 1024 / 1024
                    quant_mb = quant_path.stat().st_size / 1024 / 1024
                    print(f"  {quant_path.name}: {quant_mb:.1f} MB ({quant_mb/orig_mb*100:.0f}% of {orig_mb:.1f} MB)")
                except Exception as e:
                    print(f"  Quantization of {name} failed: {e}")
                    traceback.print_exc()

    # === Part 4: Verify ===
    print("[glirel-export] Verifying with ONNX Runtime...")
    try:
        import onnxruntime as ort

        if enc_path and enc_path.exists():
            sess_enc = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
            print(f"  Encoder inputs: {[(i.name, i.shape) for i in sess_enc.get_inputs()]}")
            print(f"  Encoder outputs: {[(o.name, o.shape) for o in sess_enc.get_outputs()]}")

        if score_path and score_path.exists():
            sess_score = ort.InferenceSession(str(score_path), providers=["CPUExecutionProvider"])
            print(f"  Scoring inputs: {[(i.name, i.shape) for i in sess_score.get_inputs()]}")

            score_feed = {
                "word_rep": score_dummy[0].numpy(),
                "word_mask": score_dummy[1].numpy(),
                "span_idx": score_dummy[2].numpy(),
                "relations_idx": score_dummy[3].numpy(),
                "rel_type_rep": score_dummy[4].numpy(),
            }
            result = sess_score.run(None, score_feed)
            max_diff = np.abs(pt_scores - result[0]).max()
            print(f"  Scoring head max diff (PT vs ONNX): {max_diff:.6f}")
            if max_diff < 1e-3:
                print("[glirel-export] Scoring head verification PASSED")

    except Exception as e:
        print(f"[glirel-export] Verification failed: {e}")
        traceback.print_exc()

    # Summary
    print()
    print("[glirel-export] === Summary ===")
    for fpath in sorted(out.iterdir()):
        if fpath.name.startswith("_"):
            continue
        size_mb = fpath.stat().st_size / 1024 / 1024
        if size_mb > 0.01:
            print(f"  {fpath.name}: {size_mb:.1f} MB")
        else:
            size_kb = fpath.stat().st_size / 1024
            print(f"  {fpath.name}: {size_kb:.1f} KB")
    print(f"[glirel-export] Output directory: {out}")


def main():
    parser = argparse.ArgumentParser(description="Export GLiREL to ONNX")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--quantize", action="store_true", default=True, help="Apply INT8 quantization (default: True)")
    parser.add_argument("--no-quantize", action="store_false", dest="quantize", help="Skip quantization")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    args = parser.parse_args()
    export_glirel(args.model, args.output, args.quantize, args.opset)


if __name__ == "__main__":
    main()
