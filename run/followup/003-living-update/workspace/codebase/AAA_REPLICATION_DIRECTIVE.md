# REPLICATION DIRECTIVE (read this first)

This file was added by the replication orchestrator (not part of the original repo).
It gives you binding guidance for reproducing this paper. The rest of this repository
(official `function_vectors` release) remains the ground truth for prompt construction,
dataset splits/filtering, seeds, and evaluation logic.

## 1. Intervention machinery: use `nnsight`
Implement the activation extraction and the function-vector (FV) interventions with the
**nnsight** package (`pip install nnsight`, runs fully locally, no remote), NOT the
repo's baukit `TraceDict` hooks — while reproducing the paper's numbers.
- Load with `nnsight.LanguageModel("EleutherAI/gpt-j-6b", device_map=...)`.
- Per-head attention outputs: read the INPUT to the attention out-projection
  (`model.transformer.h[L].attn.out_proj.input`), reshape to (n_heads, head_dim),
  following the repo's `src/utils/extract_utils.py` decomposition.
- Add the FV to the residual stream at the LAST token by editing
  `model.transformer.h[L].output[0][:, -1]` inside a trace; during multi-token
  natural-text generation, add it at every generated token (as the repo does).
- Port the repo's logic for prompts/CIE/eval/seeds/filtering exactly; only the
  tracing/patching should be nnsight. If one component is disproportionately hard in
  nnsight (e.g. the exact CIE patching loop), you may run the repo's baukit code for
  that piece and cross-check a subset with nnsight — but DOCUMENT which path produced
  each number.

## 2. Models — HF gating constraint
The environment's HF token currently has NO access to gated Llama-2 (verified 401).
- **Primary model = `EleutherAI/gpt-j-6b`** (ungated, cached in HF_HOME): use it for
  ALL headline claims (head localization, causal task triggering, per-task zero-shot
  FV accuracy, layer dependence, portability, vocab decoding).
- For the cross-model claim (paper Table 2 / Fig 4), use the ungated
  **`EleutherAI/gpt-neox-20b`** (cached) as the second model.
- Do NOT attempt any `meta-llama/Llama-2-*` model — downloads will 401. Treat the
  Llama-2 rows as an environmental limitation, not a paper failure, and say so.

## 3. Scope priority (target ~8-10 GPU-h on ONE GPU)
1. Head localization (AIE top-10 heads) + FV construction on GPT-J, 6 core tasks.
2. Causal task triggering: zero-shot + shuffled-label eval at layer 9 and the layer
   sweep, >=3 seeds, with the paper's correct-10-shot-ICL filtering applied.
3. Per-task zero-shot FV accuracy (antonym flagship etc.).
4. Portability (templates + natural text) and vocab-decoding controls if time.
5. Cross-model spot-check on GPT-NeoX-20B. Composition (Table 7) is NOT in the repo —
   reimplement only if hours remain, else descope with a note.

Datasets all ship in `dataset_files/` — no downloads. Fix seeds; apply the filter set.
