# 002-theory-update — How function-vector behavior scales with model era, family, and architecture

**Depends on:** `followup/001-living-update/results/` (per-model JSON + head sets). **0 GPU-h — analysis only.**
**Invoke:** `python -m veritas.cli.main followup <run_dir> --instruction-file <this file> --name theory-update`
(runs after 001; reuses 001's `results/`).

## Anti-goals (same as 001)
- No new GPU jobs; consume 001's on-disk artifacts only.
- Every number traced to a 001 file; nothing from memory or the paper presented as newly measured.
- Disclose every gap (missing model, antonym-only NeoX rung, base-vs-instruct confound) in `notes`.
- Do not background-and-wait; do not modify the run dir outside `followup/002-theory-update/`.

## Question
Given the 2021→2025 ladder from 001, how do **(i) FV effect size** (mean ZS FV accuracy, and its gap over the
no-FV baseline), **(ii) best/peak edit layer as a fraction of model depth**, and **(iii) head-localization
concentration** (fraction of top-AIE heads in the first ⅔ of layers; induction-head overlap) **vary with
scale, family, generation, and the MHA→GQA shift**? Fit a simple relation and restate the paper's claim in
today's terms.

## Inputs (from 001)
- Per-model: mean & per-task ZS/SL FV top-1 accuracy, baseline no-FV, peak layer + peak-layer-as-frac-depth,
  top-K AIE head list with layer indices, concentration metric, prefix-match scores, K, T used, cost note.
- GPT-J anchor read from the original replication (55.0 % on-disk); NeoX antonym-only rung (57.1 %).

## Analyses (~5)
1. **Effect size vs era/scale**: plot mean ZS FV accuracy and the +FV-minus-baseline gap vs release year and
   vs params; report whether the FV effect *strengthens, holds, or decays* across generations, split by family.
2. **Best layer vs depth**: regress peak edit layer on total depth across all rungs — fit **peak_layer ≈ α·L
   (+ β)** (expect α ≈ ⅓ per the paper's canonical |L|/3). Report α, R², residuals per model; flag any model
   whose peak deviates (e.g. deep GQA models, or the instruct rung).
3. **Localization concentration vs family/architecture**: is the "top-K heads cluster in early-mid layers"
   signature preserved? Compare concentration (frac in first ⅔) and mean layer-fraction across MHA (GPT-J,
   Llama-2, OLMo-2) vs GQA (Qwen2.5, Qwen3, Llama-3.1) models. Test whether GQA shifts localization.
4. **Induction-head overlap trend**: does the top-AIE ∩ induction-head relationship (C13) hold across
   families, or does it weaken/strengthen with generation / post-training (cf. "Thinking Sparks" emergent
   heads)? Report prefix-match of top-K heads per model.
5. **#heads sufficiency vs scale**: does the ~10-head plateau (C16) scale with K = f(n_heads), or do larger
   models need proportionally more/fewer heads? Use 001's numheads curves.

## Fitted relation (headline deliverable)
Report the peak-layer-vs-depth fit **peak_layer ≈ α·L + β** with α, β, R² and a one-line reading (e.g. "the
task-triggering layer tracks ~⅓ of network depth across four families and the MHA→GQA shift, α=…"). If effect
size vs scale is monotone, add a second short fit / trend statement.

## Updated claim in words
One paragraph restating Todd et al.'s core finding for 2025 models: e.g. "Function vectors remain a real,
transferable, causal object across families and the MHA→GQA transition; the triggering layer scales with depth
(≈α·L), the top-AIE heads stay concentrated in early-mid layers, and induction-head overlap [persists /
weakens]. The effect [holds / strengthens / decays] with generation, with [caveats: instruct confound on the
Llama-3.1 rung, antonym-only NeoX, models excluded for decoupled head_dim]."

## Deliverables (`followup/002-theory-update/`)
- `report.md` — the analyses, the fits (α, β, R²), the updated claim, and a limitations block naming every
  gap inherited from 001.
- `followup_summary.json` — the fitted coefficients, per-model derived quantities, provenance to 001 files.
- `result_card.json` (`sai.followup.result_card/v1`) — headline = the fitted relation with its key number
  (α and R²); status; 3–5 metrics (each citing the 001 artifact as baseline/provenance); ≤2 tables; 0–3
  figures under `results/` (effect-size-vs-era, peak-layer-vs-depth fit, concentration-vs-family); `notes`
  with all caveats.
- Figures reuse/extend 001's over-time and peak-layer-vs-depth plots; add a concentration-vs-family panel.

## Length target
~60–100 lines of report body; keep the fit and the one-paragraph updated claim front and center.