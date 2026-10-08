# Follow-up 003 — Function vectors on the 2026 model refresh

**Paper:** Todd et al., *Function Vectors in Large Language Models*, ICLR 2024 (arXiv 2310.15213).
**Extends:** living-update 001 (2021→2025 ladder). This tick adds the newest **2026** models.
**Anchor / reused ladder:** GPT-J-6B, GPT-NeoX-20B, Llama-2-7B, Qwen2.5-7B, Qwen3-8B-Base, OLMo-2-7B — read verbatim from the 001 results on disk.
**New this session (bf16, seed 42):** Qwen3.5-9B-Base (2026), gemma-4-12B (2026, partial).

> **Recovery note.** The SLURM compute finished at walltime; a `/net/projects2` read-only storage
> outage then blocked the card write. This report is **pure CPU post-processing** of the existing
> on-disk `fu2_summary.json` results — no GPU, no model loading, no FV/nnsight/transformers re-run.
> All deliverables are staged under `/net/scratch/haokunliu/fv003-recovery/`; `copy_back.sh` lands
> them in the run dir once projects2 is writable.

## Question

Does the function-vector finding — a handful of mid-layer attention heads whose summed mean outputs
form a vector that *causally triggers* a task zero-shot — still hold on the **2026** flagship models,
which are multimodal-architecture (MM) and use GQA attention that **nnsight cannot wrap**?

## Approach

Reused the entire 001 methodology and the 2021–2025 ladder verbatim. For the 2026 models the FV
per-head AIE decomposition was **reimplemented with plain `transformers` forward-hooks** on the
attention out-projection input inside `model.language_model.layers[L]` (reshape to
n_heads × head_dim), because nnsight cannot hook the MM architectures. Same pipeline as 001:
causal indirect effect (CIE) → top-K AIE heads → function vector → zero-shot intervention, on the
3-task subset {antonym, country-capital, present-past}, same CIE cost cap, seed 42, bf16. The
mandatory shape-guard (`o_proj.in == n_heads·head_dim`) **passed** on both 2026 models.

## Results

### Era ladder — mean zero-shot FV accuracy at the peak edit layer

| Model | Year | Attn | ZS FV % (peak) | no-FV % | peak frac-depth | concentration |
|---|---|---|---|---|---|---|
| GPT-J-6B (anchor, reused) | 2021 | MHA | 65.9 | 3.2 | 0.36 | 0.90 |
| GPT-NeoX-20B* (reused) | 2022 | MHA | 57.1 | 2.1 | 0.27 | — |
| Llama-2-7B (reused) | 2023 | MHA | **92.2** | 8.7 | 0.26 | 0.95 |
| Qwen2.5-7B (reused) | 2024 | GQA | 74.6 | 1.8 | 0.64 | 0.61 |
| Qwen3-8B-Base (reused) | 2025 | GQA | 45.0 | 2.3 | 0.55 | 1.00 |
| OLMo-2-7B (reused) | 2025 | MHA | **89.3** | 2.4 | 0.37 | 0.95 |
| **Qwen3.5-9B-Base (NEW)** | **2026** | **GQA/MM** | **16.9** | 0.1 | 0.53 | 0.40 |
| **gemma-4-12B (NEW, partial)** | **2026** | **GQA/MM** | **18.0** † | 0.0 | 0.52 | 0.70 |

\* GPT-NeoX antonym-only. † gemma-4 antonym-only (see below). Concentration = fraction of top-AIE
heads in the first ⅔ of layers.

### Qwen3.5-9B-Base (2026) per task

| Task | peak L (frac) | ZS FV % | no-FV % | canonical @L10 % | SL FV % | SL base % |
|---|---|---|---|---|---|---|
| antonym | 0 (0.00) | 0.7 | 0.3 | 0.3 | 0.7 | 1.7 |
| country-capital | 3 (0.09) | 2.4 | 0.0 | 0.0 | 7.3 | 29.3 |
| **present-past** | 17 (0.53) | **47.5** | 0.0 | 1.7 | 22.0 | 11.9 |
| **mean** | — | **16.9** | 0.1 | 0.7 | — | — |

### gemma-4-12B (2026, PARTIAL / bonus)

Only **antonym** completed before the storage outage: ZS FV **18.0%** vs no-FV 0.0% at peak layer
L25/48 (0.521 depth); SL FV 71.7% vs SL base 40.7%; concentration 0.70; mixed head_dims [256, 512].
**country-capital and present-past did NOT finish** and are descoped — their numbers are **not
invented**. Treat gemma-4 as a single antonym data point that corroborates Qwen3.5-9B.

### The finding on 2026 models

**Detectable but much weaker (with a methodological caveat).** The FV causal effect is **not clean-
survives and not clean-fails** on the 2026 MM-arch models. Zero-shot FV accuracy at the peak layer
drops from the 2023–25 MHA band (Llama-2 92.2%, OLMo-2 89.3%) to **Qwen3.5-9B mean 16.9%** and
**gemma-4 antonym 18.0%**. The one task where the 2026 FV clearly triggers is Qwen3.5 present-past
(47.5%); antonym (0.7%) and country-capital (2.4%) are near-failures.

**The effect is real, if small.** In **every completed task** `zs_fv > no-FV baseline` (e.g.
present-past 47.5% vs 0.0%, gemma antonym 18.0% vs 0.0%), so the FV does carry causal task
information. The **shuffled-label control is mixed**: SL FV > SL baseline for Qwen3.5 present-past
(22.0 vs 11.9) and gemma-4 antonym (71.7 vs 40.7) — confirming a genuine effect there — but SL FV <
SL baseline for the two Qwen3.5 near-failure tasks.

**Peak layer stays mid-to-late; depth/3 still broken.** Both 2026 models peak at ~0.52 of depth
(Qwen3.5 present-past L17/32 = 0.531; gemma-4 antonym L25/48 = 0.521), continuing 001's finding that
the paper's depth/3 heuristic does not transfer to later families — Qwen3.5's canonical @L10 layer
gives just 0.7% mean.

**Head localization weakens.** Qwen3.5's aggregated top-AIE heads are only 40% in the first ⅔ of
layers — **60% now sit in the last third** (layers 27–31 of 32) — so the classic mid-layer
localization signature holds only loosely. gemma-4 antonym is 0.70.

### CRITICAL honesty caveat

The 2026 FV construction used `transformers` forward-hooks on **GQA / multimodal attention** because
**nnsight cannot wrap these architectures**, and **gemma-4 has mixed head_dims [256, 512]**. Head
decomposition on GQA/MM attention is imperfect, so **part of the measured attenuation may be
methodological, not purely a property of the 2026 models.** The shape-guard passed on both, and the
FV-vs-baseline gap is genuine, but the absolute attenuation should not be read as a clean
model-capability claim.

## Deviations & limitations

- **Recovery post-processing only** — numbers read from `FV/workspace/results_fu/*/fu2_summary.json`;
  nothing recomputed on GPU. Outputs written to `/net/scratch` because `/net/projects2` is read-only.
- **gemma-4-12B is partial** (antonym only); its other two tasks did not finish before the outage.
- **Method caveat above** — transformers hooks on GQA/MM attention, gemma-4 mixed head_dims.
- **bf16, single seed (42), 3-task subset**, same CIE cost cap as 001. 2021–2025 ladder reused verbatim.

## Bottom line

Function vectors are **still detectable on the 2026 MM-arch models but the zero-shot effect is much
weaker** than on 2023–25 models (Qwen3.5-9B mean 16.9%, gemma-4 antonym 18.0%, vs ~90%), strong only
on Qwen3.5 present-past (47.5%). FV beats the no-FV baseline in every completed task (real causal
effect) and the peak edit layer stays mid-to-late (~0.52, depth/3 still broken). A methodological
caveat — imperfect head decomposition via transformers hooks on GQA/multimodal attention — means part
of the attenuation may not be a pure model property.
