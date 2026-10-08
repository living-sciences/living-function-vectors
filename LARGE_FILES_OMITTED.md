# Large files omitted from this repository

To keep the repository lightweight and to avoid committing model-derived weight-like
artifacts, every PyTorch tensor file (`*.pt`) in the source run was omitted. These are all
intermediate activation and causal-indirect-effect tensors. They are not deliverables: the
headline numbers live in the committed `*.json` summaries and the figures under each
study's `results/` folder. Every tensor is deterministically regenerable by rerunning the
pipeline that produced it (same script, same seed, same model), so nothing here needs to be
obtained from an external store.

No committed file is 45 MB or larger, and no committed file is 95 MB or larger. The two
largest omitted artifacts were a gemma-4-12B mean-head-activations tensor (about 130.5 MB)
and a GPT-NeoX-20B mean-head-activations tensor (about 100.0 MB).

## Summary

| Category | Count | Total size |
|---|---|---|
| `*_mean_head_activations.pt` | 104 | part of 4.70 GB |
| `*_indirect_effect.pt` | 85 | part of 4.70 GB |
| `*_mean_layer_activations.pt` | 18 | part of 4.70 GB |
| **All omitted `*.pt`** | **207** | **4.70 GB (4,931,999,369 bytes)** |

Many of these files are near-duplicate copies: the follow-up studies each kept a working
copy of the replication codebase and its results, so the GPT-J and GPT-NeoX tensors in
particular appear more than once across `run/replication/codebase/src/results/`,
`run/followup/001-living-update/codebase/src/results/`, and
`run/followup/003-living-update/workspace/codebase/src/results/`.

## What each kind of file is, and how to regenerate it

### `*_mean_head_activations.pt` (104 files)
Per model and per task, the mean output of each attention head over the mean-activation
trials. This is the quantity that gets summed over the top attention heads to build the
function vector itself. Shapes scale with the model, so sizes range from about 11.4 MB
(Qwen3.5-9B) up to about 130.5 MB (gemma-4-12B). Representative per-model sizes:
GPT-J-6B about 44.5 MB, Llama-2-7B about 50.9 MB, Qwen2.5-7B about 34.9 MB, Qwen3-8B about
51.3 MB, OLMo-2-7B about 45.6 MB, GPT-NeoX-20B about 100.0 MB, gemma-4-12B about 130.5 MB.

Representative paths:
- `run/replication/codebase/src/results/gptneox/antonym/antonym_mean_head_activations.pt` (about 100.0 MB)
- `run/followup/003-living-update/workspace/results_fu/gemma4-12b_seed42/antonym/mean_head_activations.pt` (about 130.5 MB)
- `run/followup/001-living-update/codebase/src/results_fu/llama2-7b_seed42/antonym/antonym_mean_head_activations.pt` (about 50.9 MB)

Regenerate: rerun the mean-activation step of the pipeline for that model and task.
In the replication that is `src/compute_average_activations.py` (driven inside
`src/evaluate_function_vector.py`); in the follow-ups it is produced by `src/run_fu2.py`
(for 001 and the reused 003 codebase) or `workspace/qwen35_fv.py` (for the 2026 models in
003). Deterministic given the seed (42) and the model.

### `*_indirect_effect.pt` (85 files)
Per model and per task, the causal indirect effect tensor over the CIE trials, indexed by
layer and head (shape roughly [trials, n_layers, n_heads]). These drive the top-AIE head
selection. Small files, roughly 9 KB to 283 KB each.

Representative path:
- `run/replication/codebase/src/results/gptj_seed42/antonym/antonym_indirect_effect.pt` (about 46.5 KB)

Regenerate: rerun `src/compute_indirect_effect.py` (driven inside
`src/evaluate_function_vector.py` / `src/run_fu2.py`) for that model and task.

### `*_mean_layer_activations.pt` (18 files)
GPT-J layer-average hidden states for the layer-average baseline from section 2.1 of the
paper. About 231 KB each, under `run/.../results/gptj_avg_hs/<task>/`.

Regenerate: rerun `src/compute_avg_hidden_state.py` for GPT-J on that task.

## Models used (loaded from Hugging Face, not stored here)

The activation tensors above were produced from these models, which are loaded from the
Hugging Face hub or a local cache and are not part of this repository:
`EleutherAI/gpt-j-6b`, `EleutherAI/gpt-neox-20b`, `meta-llama/Llama-2-7b-hf`,
`Qwen/Qwen2.5-7B`, `Qwen/Qwen3-8B-Base`, `allenai/OLMo-2-1124-7B`, `Qwen/Qwen3.5-9B-Base`,
and `google/gemma-4-12B`.
