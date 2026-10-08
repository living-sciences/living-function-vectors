# function-vectors — 2026 model-refresh tick (extends the prior living-update to 2026 models)

It is 2026-09-11. The prior living-update stopped at 2025 models. ADD the newest 2026 models —
primarily **the newest Qwen (Qwen3.5-9B, Mar 2026)** — to the ladder, REUSING all prior results.
Read the ORIGINAL methodology + claim IDs + result_card format first:
  /net/projects2/chai-lab-models/haokunliu/alignment-batch/followup_plans/function-vectors/001-living-update.instruction.md

## Reuse (do NOT recompute)
Prior completed living-update with all its per-model results is at:
  /net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/function-vectors/run/followup/001-living-update
Copy/symlink its `results/` into your study `results/`, copy its `workspace/` scripts, and treat every
model it already covered as DONE. You ONLY compute the NEW 2026 models below, then re-aggregate ALL
(old + new) and extend the over-time figure to the 2026 point.

## New 2026 models to add
- **Qwen3.5-9B-Base (base — FV is a base-model ICL phenomenon)**
- Optionally **gemma-4-12B** (base or -it per this paper's convention) if downloaded + hookable.

## Loading the 2026 models (verified 2026-09-11 — READ CAREFULLY)
The 2026 flagships are MULTIMODAL-arch and **nnsight cannot wrap them**; use plain transformers:
```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch
m = AutoModelForImageTextToText.from_pretrained(PATH, dtype=torch.bfloat16, device_map="auto").eval()
tok = AutoTokenizer.from_pretrained(PATH)
dec = m.language_model            # the text decoder; dec.layers is the residual stream (Qwen3.5-9B: 32 layers, hidden 4096)
# residual-stream activations: m(**ids, output_hidden_states=True).hidden_states  (len = n_layers+1)
# to hook/steer a layer L: register a forward hook on dec.layers[L]
# unembedding: m.get_output_embeddings()  ; input embeddings: m.get_input_embeddings()
```
Do NOT use nnsight for these models. Smoke-test load+one-forward+one-activation-read on the NEW model
BEFORE the full run; if it genuinely cannot be hooked in your pipeline after a real attempt, DESCOPE
that model with a clear note and finalize with the models that worked (adding Qwen3.5 to the ladder is
the goal; a documented block is acceptable, a silent skip is not).
2026 model paths (local, cached): Qwen3.5-9B-Base = /net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9 ; Qwen3.5-9B (instruct) = /net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
gemma-4-12B(-it) is downloading to HF_HOME (google/gemma-4-12B / google/gemma-4-12B-it) — add it as a
SECOND 2026 point IF present and hookable (same recipe, arch Gemma4UnifiedForConditionalGeneration);
skip with a note if not yet downloaded.

## Paper-specific
FV's original used nnsight for the per-head AIE decomposition; for Qwen3.5-9B you MUST reimplement the head-decomposition via transformers forward hooks on the attention out-projection input inside m.language_model.layers[L] (reshape to n_heads×head_dim). Smoke-test the shape-guard first. Apply the SAME CIE cost cap as the original. Report peak zero-shot FV accuracy + best edit layer for Qwen3.5-9B and whether the depth/family finding holds.

## Deliverables (same schema as the original)
report.md, followup_summary.json, result_card.json (sai.followup.result_card/v1) — the comparison
table + over-time figure must now include the 2026 model(s); headline should state whether the
paper's finding still holds on the 2026 models. Every number from this session's execution; original
values read from the prior results on disk. Foreground only; never background-and-yield.