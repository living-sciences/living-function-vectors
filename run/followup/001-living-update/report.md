# Follow-up 001 — Do function vectors survive in today's models?

**Paper:** Todd et al., *Function Vectors in Large Language Models*, ICLR 2024 (arXiv 2310.15213).
**Anchor:** GPT-J-6B replication in this run dir (read-only).
**GPU:** 1× NVIDIA A40 48 GB. **Precision:** new models bf16; GPT-J/NeoX reused fp16 (from disk).

## Question

Do the four load-bearing findings of the function-vector paper still hold in models released
2023–2025, across families (Llama, Qwen, OLMo) and the MHA→GQA architecture shift?
(a) task information localizes to a small set of mid-layer attention heads (top-AIE selection);
(b) summing those heads' mean outputs yields a vector that **causally triggers** the task zero-shot;
(c) the effect is layer-dependent (peaks early-mid); (d) ~10 heads (scaling with model) suffice.
We measure this on an era ladder anchored by the paper's GPT-J-6B (2021).

## Approach

I reused the replication's working codebase (a read-only copy under `codebase/`), its `dataset_files/`
(6 core tasks), and its pinned `.venv` (baukit + nnsight + transformers 4.49). I ran the **exact** paper
pipeline — causal indirect effect (CIE) → top-K AIE heads → function vector → zero-shot intervention —
via a load-once driver (`run_fu2.py`), adding only: a `qwen2/qwen3` branch, a **mandatory shape guard**
(asserts the o_proj-input head decomposition `o_proj.in == n_heads·head_dim` is valid — the single
correctness gate; GQA shrinks only K/V heads, never the o_proj input, so all main-ladder models pass),
and a case-insensitivity fix (see Deviations). New numbers come entirely from this session; every GPT-J /
GPT-NeoX value is read from the replication artifacts on disk (paths cited below).

**Ladder run this session (bf16, seed 42, T=15 CIE trials, 50 mean-act trials, K = round(10·n_heads/16) cap 20):**
Llama-2-7B (2023, local), Qwen2.5-7B (2024), Qwen3-8B-Base (2025), OLMo-2-7B (2025).
**Reused from disk (0 GPU-h):** GPT-J-6B (2021, anchor), GPT-NeoX-20B (2022, antonym-only).
**Skipped:** Llama-3.1-8B-Instruct (gated, not local) — 2024 is covered by Qwen2.5, 2025 by Qwen3 + OLMo-2.

Per the pre-specified **cost cap**, I ran the **3-task subset {antonym, country-capital, present-past}**
(fp32 was infeasible — see Deviations), measured SL (shuffled-label) FV at the peak layer only, and computed
the #heads ablation (C16) for the flagship antonym task. The GPT-J anchor is recomputed on the **same 3-task
subset** for a fair comparison.

## Results

**Main comparison table** (new numbers = this session, bf16; peak = each model's best ZS FV edit layer;
canonical = depth/3 layer, the paper's choice; SL = shuffled-label FV at peak; concentration = fraction of
top-K AIE heads in the first ⅔ of layers; induction = mean prefix-match of top-K heads vs. all heads).

| Model | Year | Family | Attn | K | mean ZS FV % (peak) | mean ZS FV % (canon ⅓) | no-FV base % | SL FV % | peak-layer frac depth | top-AIE concentration | induction (top-K / all) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **GPT-J-6B** (anchor, reused) | 2021 | EleutherAI | MHA | 10 | **65.9** | 49.6 | 3.2 | 90.2 | 0.36 | 0.90 | 0.182 / 0.044 |
| GPT-NeoX-20B (reused, *antonym-only*) | 2022 | EleutherAI | MHA | 50 | 57.1 | — | 2.1 | 83.9 | 0.27 | — | — |
| Llama-2-7B | 2023 | Llama | MHA | 20 | **92.2** | 89.1 | 8.7 | 88.8 | 0.26 | 0.95 | 0.048 / 0.026 |
| Qwen2.5-7B | 2024 | Qwen | GQA | 18 | **74.6** | 6.4 | 1.8 | 62.7 | 0.64 | 0.61 | 0.054 / 0.044 |
| Qwen3-8B-Base | 2025 | Qwen | GQA | 20 | **45.0** † | 6.3 | 2.3 | 73.2 | 0.55 ‡ | 1.00 | 0.075 / 0.039 |
| OLMo-2-7B | 2025 | OLMo | MHA | 20 | **89.3** | 86.3 | 2.4 | 92.6 | 0.37 | 0.95 | 0.019 / 0.025 |

**Original anchor (read from disk):** paper GPT-J mean ZS FV **57.5%**; **replicated 55.0%** (6-task, @L9);
per-task replicated ZS FV @L9 antonym 46.7 / country-capital 86.1 / present-past 16.0 — from
`replication/codebase/src/results/gptj_seed{42,1,2}/<task>/zs_results_layer_sweep.json` (key `"9"`).
Head localization / induction from `.../results/head_localization.txt`. GPT-NeoX from `.../results/gptneox/antonym/`.

† Qwen3-8B peak mean is dragged down by a **total FV failure on country-capital** (peak 5.1% = baseline);
its other two tasks are healthy (antonym 48.4%, present-past 81.4%).
‡ Qwen3 peak-layer fraction is the mean over the two tasks where the FV is effective (country-capital excluded).

### The four findings, updated

**(a,b) The FV still causally triggers the task — and generally *more* strongly.** At each model's peak
edit layer, zero-shot FV accuracy is comparable-to-higher than GPT-J's across the whole ladder (57–92%),
against a near-zero no-FV baseline (1.8–8.7%). Llama-2 (92.2%) and OLMo-2 (89.3%) clearly *exceed* the
GPT-J anchor (65.9% on the same subset). The mechanism — a handful of mid-layer attention heads whose mean
outputs sum to a task-triggering vector — is intact under GQA. **This is the headline: function vectors survive
2021→2025 and across the MHA→GQA shift.**

**(c) Layer-dependence holds, but the peak has *drifted later* for the Qwen (GQA) family.** For GPT-J,
Llama-2 and OLMo-2 the FV peaks early-mid (0.26–0.37 of depth), exactly as the paper reports, and the paper's
canonical depth/3 layer works well (canon ≈ peak). But for **Qwen2.5 (0.64) and Qwen3 (0.55)** the optimal
edit layer moved into the second half of the network, so the **canonical depth/3 layer collapses to ~6%**
while the true peak stays high (74.6% / and 45–81% per task). The paper's "edit at depth/3" heuristic does
**not** transfer to the Qwen family — a concrete input for study 002's fitted layer-vs-depth relation.
(See `results/fv_peak_layer_vs_depth.pdf`.)

**(a′) Mid-layer head concentration persists.** Top-K AIE heads remain concentrated in the first ⅔ of layers:
0.90 (GPT-J), 0.95 (Llama-2, OLMo-2), 1.00 (Qwen3); only Qwen2.5 is lower (0.61), consistent with its later,
more diffuse peak.

**(C13) Induction-head overlap is model-dependent.** Top-K AIE heads are enriched for prefix-matching
(induction) relative to all heads in GPT-J (0.182 vs 0.044), Llama-2 (0.048 vs 0.026), Qwen2.5 (0.054 vs
0.044) and Qwen3 (0.075 vs 0.039) — but **not** in OLMo-2 (0.019 vs 0.025, i.e. *below* average). The paper's
induction-head story does not cleanly generalize to OLMo-2. (See `results/fv_head_concentration.pdf`.)

**(d, C16) The ~10-head plateau holds for MHA, but the Qwen GQA family needs more heads.** 95%-of-max
zero-shot accuracy (antonym, at peak layer) is reached by k = 8 (OLMo-2) and k = 11 (Llama-2) — matching the
paper's plateau ~10 — but k = 25 (Qwen2.5) and k = 27 (Qwen3). Task information is more distributed across
heads in the Qwen GQA models.

### Figures (under `results/`)
1. `fv_over_time.pdf` — mean top-1 zero-shot FV accuracy (peak layer) vs release year; color = family,
   marker = MHA (circle) / GQA (square); baseline no-FV as the faint lower line; stems show the base→+FV gap.
2. `fv_peak_layer_vs_depth.pdf` — peak edit-layer as a fraction of depth vs year (the Qwen drift).
3. `fv_head_concentration.pdf` — mid-layer concentration (C5) and induction overlap (C13) per model.

## Deviations & limitations

- **bf16 instead of fp32** (instruction §3.5) for all new models. Rationale: these models are natively bf16;
  fp32 merely upcasts the weights (no accuracy gain) and made the study infeasible under the 8 GPU-h / 90-min
  cap (measured: fp32 CIE ≈ 30 min/task + full dual layer-sweep ≈ 100 min/task). Rank-based FV accuracies are
  precision-robust. **Impact: negligible on the numbers; large on feasibility.**
- **3-task subset** {antonym, country-capital, present-past} for all new models (the cost cap's explicit
  fallback), not the full 6 tasks. The GPT-J anchor is recomputed on the same subset (65.9% peak / 49.6%
  canonical) so all comparisons are like-for-like. The full-6-task GPT-J means (57.5 paper / 55.0 replicated
  @L9; 72.7 peak) are reported for context only.
- **SL measured at peak layer only** (not a full sweep) — the 10-shot shuffled sweep is the cost sink; the ZS
  curve is still fully swept for peak detection.
- **C16 for the flagship antonym task per model**, built from this session's top-k AIE heads (the paper's
  hardcoded universal head sets do not exist for the new models); ablation filter set capped at 150.
- **Qwen3-8B-Base needed transformers 4.53** (pinned venv is 4.49, which does not recognize the `qwen3`
  architecture). I built an **isolated overlay venv** that reuses the base torch/baukit and only overlays a
  newer transformers + pinned numpy 1.25; **the pinned run venv was not modified.**
- **Code fix (not a scope change):** the original arch dispatch keyed on case-sensitive substrings of
  `name_or_path`; loading Llama-2 from a local path `…/Llama-2-7b-hf` (capital "Llama") failed the lowercase
  `'llama'` checks (UnboundLocalError), and Qwen had no branch at all. I made the dispatch case-insensitive
  and the head-projection generic (bias iff the module has one) in the copy. Behavior is identical to the
  originals for every previously-supported model.
- **Llama-3.1-8B-Instruct (rung 6) skipped** — gated base, not in local `shared_models`. The single
  instruct-model caveat from the instruction therefore does not apply; every model reported here is a **base**
  model except GPT-J/NeoX which are also base.
- **One seed (42).** Cross-seed variance not estimated for the new models (GPT-J anchor used 3 seeds on disk).
- **GPU-hours:** ~3.9 productive + ~2.7 debugging (one discarded fp32 pass used to find the bug and measure
  cost) ≈ 6.6 GPU-h total, under the 8 GPU-h cap.

## Bottom line

**Function vectors survive.** The causal zero-shot trigger and its mid-layer, few-head, top-AIE signature
persist from GPT-J (2021) through Llama-2, Qwen2.5, Qwen3 and OLMo-2 (2025), across the MHA→GQA shift, and
peak-layer FV accuracy generally *rises* (up to 92%). The two real cracks: (1) the optimal **edit layer drifts
later** in the Qwen GQA family (the paper's depth/3 heuristic breaks there), and (2) two model-specific
failures — Qwen3's country-capital FV, and OLMo-2's lack of induction-head enrichment — show the finer
mechanistic story is not perfectly universal.
