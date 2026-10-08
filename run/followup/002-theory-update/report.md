# Follow-up 002 — How function-vector behavior scales with era, family, and the MHA→GQA shift

**Paper:** Todd et al., *Function Vectors in Large Language Models*, ICLR 2024 (arXiv 2310.15213).
**Depends on:** `followup/001-living-update/results/` (per-model JSON + head sets) and the replication
anchors on disk. **0 GPU-h — analysis only.** No model was run; every number is read from a 001 or
replication artifact and re-derived arithmetically here.

## Question

Given 001's 2021→2025 era ladder (GPT-J-6B → GPT-NeoX-20B → Llama-2-7B → Qwen2.5-7B → Qwen3-8B-Base →
OLMo-2-7B), how do **(i) FV effect size** (mean zero-shot FV accuracy and its gap over the no-FV baseline),
**(ii) peak edit layer as a fraction of depth**, and **(iii) head-localization concentration** (top-AIE
heads in the first ⅔ of layers; induction-head overlap) vary with **scale, family, generation, and the
MHA→GQA shift**? Fit a simple relation (`peak_layer ≈ α·L + β`) and restate the paper's claim in today's
terms.

## Approach

Pure re-analysis of 001's committed artifacts. I read the four new models' per-model quantities from
`001/results/per_model/consolidated.json` and the `head_localization.json` / `antonym_perf_v_heads.json`
files, and the two anchors (GPT-J, GPT-NeoX) from 001's committed values, **cross-checked against the
replication disk**: recomputing GPT-J's peak from `replication/.../gptj_seed{42,1,2}/<task>/zs_results_layer_sweep.json`
by per-seed argmax then averaging reproduces 001's 65.9 % peak and 0.361 peak-fraction exactly (baseline
within 0.3 pp). I assembled a 6-rung master table (`results/derived_master.json`), fit `peak_layer = α·L + β`
by OLS (and through-origin) over all rungs and over the MHA subset, and computed effect-size-vs-era/scale
trends, concentration by architecture, induction enrichment ratios, and the #heads plateau as a fraction of
`n_heads`. Figures reuse 001's palette/markers (color = family, ○ = MHA, ■ = GQA). "Effective" peak layer
excludes tasks whose FV failed (Qwen3 country-capital, whose spurious layer-0 "peak" is not a real trigger).

## Results

### Master era ladder (all quantities traced to 001 / replication)

| Model | Year | Attn | L | n_heads | peak layer (eff) | peak frac | mean ZS FV % | no-FV % | gap (pp) | concentration | mean layer-frac | induction top-K/all | plateau k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GPT-J-6B** (anchor) | 2021 | MHA | 28 | 16 | 10.1 | 0.361 | 65.9 | 3.2 | 62.7 | 0.90 | — | 0.182 / 0.044 | 10 |
| GPT-NeoX-20B (antonym-only) | 2022 | MHA | 44 | 64 | 12.0 | 0.273 | 57.1 | 2.1 | 55.0 | — | — | — | — |
| Llama-2-7B | 2023 | MHA | 32 | 32 | 8.3 | 0.260 | 92.2 | 8.7 | 83.5 | 0.95 | 0.436 | 0.048 / 0.026 | 11 |
| Qwen2.5-7B | 2024 | **GQA** | 28 | 28 | 18.0 | 0.643 | 74.6 | 1.8 | 72.8 | 0.61 | 0.663 | 0.054 / 0.044 | 25 |
| Qwen3-8B-Base | 2025 | **GQA** | 36 | 32 | 19.5 | 0.542 | 45.0 † | 2.3 | 42.7 | 1.00 | 0.578 | 0.075 / 0.039 | 27 |
| OLMo-2-7B | 2025 | MHA | 32 | 32 | 11.7 | 0.365 | 89.3 | 2.4 | 86.9 | 0.95 | 0.514 | 0.019 / 0.025 | 8 |

† Qwen3 mean is dragged down by a total FV failure on country-capital (5.1 % = baseline); its other two
tasks are healthy (antonym 48.4 %, present-past 81.4 %). Its "effective" peak layer/fraction excludes that
failed task.

### Fitted relation — HEADLINE deliverable: `peak_layer ≈ α·L + β`

- **MHA subset (GPT-J, NeoX, Llama-2, OLMo-2), through origin: α = 0.30** — i.e. `peak_layer ≈ 0.30·L`,
  essentially the paper's canonical `|L|/3`. The per-model residuals are within **±1.9 layers** of the L/3
  line, and the peak-fraction is a tight cluster: **0.315 ± 0.048** across three families and 2021→2025.
  (OLS on the 4-point MHA set gives α = 0.135, β = 5.94, R² = 0.31; the low R² is an artifact of the narrow
  MHA depth range, 28–44 layers — the robust statement is the tight peak-fraction cluster, not the slope.)
- **All 6 rungs, single slope: the fit collapses — α = 0.02, β = 12.6, R² = 0.001** (through-origin
  α = 0.39, R² < 0). A single depth-proportional rule does **not** span the ladder, because the **GQA Qwen
  family breaks it**: Qwen2.5 peaks at 0.643·L and Qwen3 at 0.542·L, a peak-fraction of **0.593 ± 0.05 —
  roughly twice the MHA value**. Editing at `L/3` therefore collapses to ~6 % accuracy for the Qwen models
  (001's canonical-layer numbers) while their true peak stays high (74.6 % / 45–81 %).

**Reading:** *the task-triggering layer tracks ≈⅓ of network depth (α ≈ 0.30) across four MHA models spanning
three families and 2021→2025 — the paper's `|L|/3` heuristic holds for MHA — but the MHA→GQA shift moves the
trigger to ≈0.6·L in the Qwen family, so no single α fits the whole ladder (α=0.02, R²=0.00).*

### (1) Effect size vs era / scale

FV **holds high and does not follow a monotone era or scale trend.** Mean ZS FV at the peak layer stays in
the **45–92 %** band 2021→2025 against a near-zero (1.8–8.7 %) no-FV baseline, so the **+FV-minus-baseline
gap is large everywhere (43–87 pp)**. The year trend is flat (slope +1.0 %/yr, R² = 0.01) and the scale
trend is weakly *negative* (FV vs log₁₀ params: r = −0.44, R² = 0.19 — the 20 B NeoX and the largest new
models are not the strongest). Effect size is governed by **family / pre-training, not by generation or
size**: the strongest are Llama-2 (92.2 %) and OLMo-2 (89.3 %), both MHA; the weakest new rung is Qwen3
(45.0 %, pulled down by its one task failure). **Verdict: the FV effect *holds* (neither strengthens nor
decays with era); it is real and large across every family and across the MHA→GQA shift.**

### (2) Best layer vs depth — see fit above and `results/fig2_peak_layer_vs_depth_fit.png`.

### (3) Localization concentration vs family / architecture

The "top-K heads cluster in early-mid layers" signature is **preserved but shifted deeper under GQA.** Every
model keeps a majority of its top-AIE heads in the first ⅔ of layers (concentration 0.61–1.00). Mean
concentration is **MHA 0.93 vs GQA 0.81**, and the mean layer-fraction of the top-K heads is **MHA 0.475 vs
GQA 0.621** — GQA heads sit deeper, consistent with the later peak. Within GQA the two Qwen models diverge:
Qwen3 is actually the most concentrated model on the ladder (1.00), while Qwen2.5 is the least (0.61, the
only sub-two-thirds… still 0.61 in first ⅔). **GQA shifts localization later/deeper but does not destroy it.**

### (4) Induction-head overlap trend

Top-AIE heads are enriched for prefix-matching (induction) over the all-head average in **every model except
OLMo-2**, but the enrichment **weakens sharply after GPT-J**: ratio (top-K / all) = **GPT-J 4.18**, Qwen3
1.95, Llama-2 1.88, Qwen2.5 1.23, **OLMo-2 0.74** (below average — no enrichment). The C13 induction↔FV
relationship therefore **holds directionally for four of five families but decays from the 2021 anchor and
breaks entirely for OLMo-2**; it does not strengthen with generation or post-training on this ladder.

### (5) #heads sufficiency vs scale

The paper's ~10-head plateau **holds for MHA but not GQA.** 95 %-of-max antonym accuracy is reached by
**k = 8 (OLMo-2), 10 (GPT-J), 11 (Llama-2)** — the ~10-head plateau, and a *small* fraction of heads
(k/n_heads 0.25–0.63) — but by **k = 25 (Qwen2.5) and k = 27 (Qwen3)**, which is **0.84–0.89 of all heads**.
The GQA Qwen models need **proportionally far more** heads (≈2.5× the count, and a much larger fraction),
i.e. their task information is markedly more distributed. Sufficiency does **not** scale as a fixed ~10;
it tracks the architecture, jumping under GQA.

## Updated claim in words

**Function vectors remain a real, transferable, causal object across model families and the MHA→GQA
transition.** Summing a handful of top-AIE attention heads' mean outputs still triggers the task zero-shot
from a near-zero baseline in every 2021→2025 model tested (peak accuracy 45–92 %, gap 43–87 pp), and the
effect **holds** with generation rather than strengthening or decaying — it is set by family/pre-training,
not by year or parameter count. **For MHA models the triggering layer still scales with depth as ≈⅓·L
(α = 0.30, peak-fraction 0.315 ± 0.05 across three families), confirming Todd et al.'s `|L|/3` heuristic**;
the top-AIE heads stay concentrated in the early-mid layers (0.90–1.00 in the first ⅔). **But the MHA→GQA
shift moves the causal trigger later (to ≈0.6·L in the Qwen family) and spreads it across many more heads
(≈25–27 vs ≈10), so the single depth-proportional rule breaks across the full ladder (α = 0.02, R² = 0.00)
and editing at `L/3` fails for GQA.** Induction-head overlap **persists directionally but weakens** after
GPT-J and disappears for OLMo-2. Caveats: the Qwen3 country-capital FV fails entirely; the NeoX rung is
antonym-only; no instruct rung was measured (001's Llama-3.1-8B-Instruct was gated/skipped, so the
base-vs-instruct confound in the instruction could not be tested); all new numbers are single-seed bf16
inherited from 001.

## Deviations & limitations (every gap inherited from 001, plus 002's own choices)

- **No instruct rung / base-vs-instruct confound untestable.** 001 skipped Llama-3.1-8B-Instruct (gated,
  not local), so the ladder is all *base* models except none-instruct. The instruction's expected
  "instruct rung on Llama-3.1" deviation therefore could not be evaluated at all — flagged, not worked
  around.
- **NeoX is antonym-only** (001 reused a single-task rung): it enters the effect-size and peak-layer fits
  but has no concentration/induction/#heads data (blank cells above), and its "mean ZS FV" is one task.
- **Qwen3 country-capital FV failure** (5.1 % = baseline) is a genuine per-model crack from 001; I excluded
  that task's spurious layer-0 "peak" from Qwen3's effective peak layer (using it would have put Qwen3 at
  peak-fraction 0.36, masking the GQA drift). Both the effective (0.542) and the all-task (0.361) figures
  are in `derived_master.json`.
- **n = 6 rungs, single seed, bf16.** Family-split regressions are not meaningful (1–2 points per family);
  I report family effects descriptively. The scale/era trends have wide CIs (p ≈ 0.4–0.9) — the honest
  reading is "no monotone trend / effect holds," not a precise slope.
- **Low R² on the MHA `peak_layer ≈ α·L` fit is a range artifact,** not a failure of the L/3 rule: the four
  MHA depths span only 28–44 layers, so there is little L-variance to explain even though every MHA peak
  fraction sits at 0.26–0.37. The robust headline is the peak-fraction cluster (0.315 ± 0.05 ≈ ⅓), which
  the figure shows hugging the L/3 line.
- **Precision mismatch inherited from 001:** GPT-J/NeoX are fp16 on-disk; the four new models are bf16.
  Rank-based FV accuracies are precision-robust (001's finding), but the anchor-vs-new comparison is not
  perfectly like-for-like in dtype.
- **Nominal parameter counts** (GPT-J 6.0, NeoX 20, Llama-2 6.7, Qwen2.5 7.6, Qwen3 8.2, OLMo-2 7.3 B) are
  from model cards, used only for the scale trend; they are not re-measured.
- **Provenance:** all six rungs' source files are listed per-row in `followup_summary.json`
  (`comparison_to_original`) and in `results/derived_master.json` (`prov` field per model).
