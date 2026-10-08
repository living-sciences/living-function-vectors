# 001-living-update — Do function vectors survive in today's models?

**Paper:** Todd et al., *Function Vectors in Large Language Models*, ICLR 2024 (arXiv 2310.15213).
**Run dir (READ-ONLY replication):** `/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/function-vectors/run/`
**This study writes to:** `followup/001-living-update/` inside the run dir (created by the followup CLI).
**Invoke:** `python -m veritas.cli.main followup <run_dir> --instruction-file <this file> --name living-update`

---

## 0. HARD RULES (anti-goals — read first)

- **NEVER background a long job and end the turn to "wait for it."** Run each GPU step in the foreground of
  this session; block on it; read its output; proceed. If a step would exceed the budget, cut scope per the
  cost cap below — do not park a job and return.
- **No token budgets, no time-boxing of the reasoning.** Finish the study.
- **Every new number in the deliverables must come from THIS session's execution.** Do not copy a number from
  a paper, a memory, or a prior run and present it as newly measured.
- **All "original" GPT-J / paper numbers are READ FROM DISK** at the exact artifact paths cited in §3, not
  recomputed from memory. Cite the file for each.
- **Disclose every hole and every hand-set value.** If a model is skipped, a task subset used, or a value
  filled by hand, say so in `notes` and in the report. No silent gaps.
- **Do not modify the replication run dir.** Reuse `replication/codebase/` read-only; write only under
  `followup/001-living-update/`. Make a working copy of the codebase inside the followup dir if you must edit
  `model_utils.py` (you will — see §5).
- Storage under `/net/projects2/chai-lab-models/haokunliu/` only; `HF_HOME=…/alignment-batch/hf-cache`. Never `/home`.
- One GPU (A40 48 GB or A100 80 GB). Python via `uv`, own `.venv` under the followup dir, or reuse the run's
  `.venv` (nnsight + baukit + transformers 4.49 already pinned there — REUSE it; do not upgrade).

---

## 1. Question

Do the four load-bearing findings of the function-vector paper — (a) task information localizes to a small set
of mid-layer attention heads (top-AIE selection), (b) summing those heads' mean outputs yields a vector that
*causally triggers* the task zero-shot, (c) the effect is layer-dependent (peaks early-mid, collapses late),
and (d) a handful of heads (~10, scaling with model) suffices — **still hold in models released 2023–2025**,
across families (Llama, Qwen, OLMo) and the MHA→GQA architecture shift? We measure this on an era ladder
anchored by the paper's GPT-J-6B (2021).

---

## 2. Claims this study updates (IDs, paper value, replicated value, artifact)

Read the replicated (original) column from disk at these paths (all under
`run/replication/codebase/src/`). These become the **GPT-J baseline column** of the comparison table — use the
**replicated** values, not the paper's, as the anchor the new models are compared against (see anti-goal on
E-F below).

| ID | Quantity (per model) | Paper (GPT-J) | Replicated (GPT-J) — READ FROM | 
|----|----------------------|---------------|--------------------------------|
| **C1/C3** | mean & per-task top-1 **zero-shot FV accuracy @ canonical layer**; baseline (no-FV) | mean ZS **57.5**, SL 90.8; baseline ZS 5.5 | mean ZS **55.0**, SL 90.8, baseline ZS 5.7 — `results/gptj_seed{42,1,2}/<task>/{zs,fs_shuffled}_results_layer_sweep.json`, key `"9"`, `intervention_topk[0][1]` (FV) / `clean_topk[0][1]` (baseline) |
| **C2** | **peak edit-layer** of the ZS FV curve (per task + median) | early-mid, late cliff | e.g. antonym peak L11=0.60; late L19-27≈0.01 — same sweep JSONs, all layer keys |
| **C5** | **top-AIE head set** + their layer indices (concentration) | 9/10 in early-mid | top-10 = (15,5)(9,14)(12,10)(11,0)(8,1)(13,13)(14,9)(8,0)(24,6)(6,6) — `results/head_localization.txt` |
| **C13** | induction-head **prefix-match** of top-AIE heads | 8-1 .49, 12-10 .56, 24-6 .31 | 0.545 / 0.569 / 0.333 — `results/head_localization.txt` |
| **C16** | **#heads plateau** (accuracy vs k) | plateau ~10 | reproduced — `results/gptj_test_numheads/<task>_perf_v_heads.json` |

Per-task replicated ZS FV @ L9 (C3), for the table's GPT-J row (`gptj_seed*` seed-avg):
antonym **46.7**, capitalize **68.9**, country-capital **86.1**, english-french **70.3**, present-past **16.0**,
singular-plural **42.2**.

**GPT-NeoX-20B (2022) — reuse on-disk, do NOT recompute.** `results/gptneox/antonym/` has antonym only:
ZS peak **57.1 %** @ L12, SL peak **83.9 %** @ L12 (`{zs,fs_shuffled}_results_layer_sweep.json`). Include as a
2022 rung with an explicit "antonym-only, reused" note.

## 2b. Claims this study does NOT update (and why)

- **C4** (h̄ layer-average), **C7** (vocab reconstruction), **C8** (logit-lens decode), **C9** (natural-text
  portability): GPT-J-only controls about *why* FV works, not *whether it survives across models*. Out of
  scope for the ladder; would blow the CIE budget (C7 is gradient optimization). Leave the GPT-J replication
  values as-is; do not extend to new models.
- **C10** (template portability): optional add-on. Run only for the flagship task (antonym) on 1-2 new models
  if the wall-clock budget has slack; otherwise skip and say so.
- **C6, C11, C12, C15**: out_of_scope in the replication (viz / composition / cyclic / appendix); unchanged.
- **C14** (cross-model max-AIE trend): this study *is* the update to C14 — the paper stated it over GPT-J +
  Llama-2-7B/70B which were gated at replication time. We generalize it to the 2021→2025 ladder.

---

## 3. Exact reuse of the replicated codebase

Work from a **copy** of `run/replication/codebase/` placed under `followup/001-living-update/codebase/`
(so the run dir stays untouched). Reuse its `dataset_files/` (the 6 core tasks ship there — never re-fetch).
Reuse the run's `.venv` (baukit + nnsight + transformers 4.49 pinned).

**Scripts (unchanged) and their args:**
- **CIE + layer-sweep eval (C1/C2/C3/C5)** — `evaluate_function_vector.py` (driven by `run_step2.py`'s
  load-once pattern to avoid reloading weights per task):
  ```
  cd followup/001-living-update/codebase/src
  python evaluate_function_vector.py --dataset_name=<TASK> --model_name=<HF_ID> \
    --n_top_heads=<K> --edit_layer=-1 --seed=42 --save_path_root=results_fu/<nick>_seed42 \
    --n_indirect_effect_trials=<T> --n_mean_activations_trials=50
  ```
  Emits per task: `{zs,fs_shuffled}_results_layer_sweep.json` (keyed by layer, `intervention_topk`/`clean_topk`),
  `model_baseline.json`, `<task>_indirect_effect.pt` [T, n_layers, n_heads], `<task>_mean_head_activations.pt`,
  `fv_eval_args.txt`. **Reads intervention path = baukit** (parity with the GPT-J numbers).
- **Head localization + induction (C5/C13)** — the step-5 one-liner in
  `run/analyze/replication_plan.json` step id 5 (aggregate `*_indirect_effect.pt` → top-K AIE heads +
  layer indices; `extract_utils.prefix_matching_score(model, cfg)` for those heads). Tee to
  `results_fu/<nick>_head_localization.txt`.
- **#heads ablation (C16)** — `test_numheads.py --dataset_name=<TASK> --model_name=<HF_ID>
  --model_nickname=<nick> --n_heads=40 --edit_layer=<peak> --save_path_root=results_fu
  --mean_act_root=<nick>_seed42` (reuses the mean-acts from the CIE step; cheap). Run for ≤3 tasks/model.
- **Template portability (C10, optional)** — `portability_eval.py … --edit_layer=<peak>` antonym only.

**Per-model code changes (spell out; make them in the COPY only):**
1. **Add `qwen2` / `qwen3` branches to `src/utils/model_utils.py`.** The current `else` raises
   `NotImplementedError` for Qwen. Clone the `llama` branch, key on `'qwen' in model_name.lower()`:
   ```python
   elif 'qwen' in model_name.lower():
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       tokenizer.pad_token = tokenizer.eos_token
       model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
       MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                     "n_layers":model.config.num_hidden_layers,
                     "resid_dim":model.config.hidden_size,
                     "name_or_path":model.config._name_or_path,
                     "attn_hook_names":[f'model.layers.{L}.self_attn.o_proj' for L in range(model.config.num_hidden_layers)],
                     "layer_hook_names":[f'model.layers.{L}' for L in range(model.config.num_hidden_layers)],
                     "prepend_bos":False}
   ```
2. **Llama-3.1-8B and Llama-2-7B**: the existing `llama` branch works AS-IS (matches `'llama'`, uses
   `model.layers.*.self_attn.o_proj`, `prepend_bos=True`). Llama-2 loads from the local path
   `…/hf-cache-local/Llama-2-7b-hf`. For Llama-3.1 pass `…/shared_models/meta-llama/Llama-3.1-8B-Instruct`
   (base is gated) and **flag it as instruct** in every deliverable.
3. **OLMo-2-7B**: existing `olmo` branch works (matches `'olmo'`, standard `model.layers.*.self_attn.o_proj`).
4. **Shape guard (MANDATORY, before CIE on every model):** assert the head decomposition is valid —
   `assert model_config['resid_dim'] % model_config['n_heads'] == 0` and that the o_proj input width equals
   `n_heads * (resid_dim//n_heads)`. If it fails (Gemma-3 / gpt-oss), **skip that model** and record the skip
   in `notes`. This is the single correctness gate for the whole study (see risk #1 in research_notes).
5. Set `FV_FP16` **unset** for the new ≤9B models (load bf16 in the qwen branch / fp32 default elsewhere);
   GPT-J's on-disk numbers used fp16 — note the precision difference rather than re-running GPT-J.

---

## 4. The era ladder & per-model computation (with the cost cap)

Verified 2026-09-09 (gating via keyless HF API; arch via local `config.json`). All main-ladder models have
**head_dim = hidden/n_heads** so the o_proj-input decomposition is valid (GQA shrinks only K/V heads, never the
o_proj input — see research_notes §3).

| # | Model (year) | HF id / path | L × H (kv) | head_dim | attn | K (n_top_heads) | T (CIE trials) | source |
|---|--------------|--------------|-----------|----------|------|-----------------|----------------|--------|
| 1 | GPT-J-6B (2021) | `EleutherAI/gpt-j-6b` (cache) | 28×16 | 256 | MHA | 10 | — (reuse disk) | anchor |
| 2 | Llama-2-7B (2023) | `…/hf-cache-local/Llama-2-7b-hf` | 32×32 (32) | 128 | MHA | 20 | 15 | local |
| 3 | Qwen2.5-7B (2024) | `Qwen/Qwen2.5-7B` (base, ungated; dl ~15GB) | 28×28 (4) | 128 | GQA | 18 | 15 | download |
| 4 | Qwen3-8B (2025) | `Qwen/Qwen3-8B-Base` (ungated) / local `…/shared_models/Qwen/Qwen3-8B` | 36×32 (8) | 128 | GQA+QKnorm | 20 | 15 | local/dl |
| 5 | OLMo-2-7B (2025) | `allenai/OLMo-2-1124-7B` (base, ungated; dl ~14GB) | 32×32 (32) | 128 | MHA | 20 | 15 | download |
| (6) | Llama-3.1-8B (2024, **instruct**) | `…/shared_models/meta-llama/Llama-3.1-8B-Instruct` | 32×32 (8) | 128 | GQA | 20 | 15 | local (flag instruct) |
| (7) | GPT-NeoX-20B (2022) | reuse `results/gptneox/antonym/*` | 44×64 | 96 | MHA | (50, on disk) | — | reuse antonym only |

Run rungs 2–5 (core), optionally 6. Rungs 1 & 7 are read from disk.

**Per model (rungs 2–6):**
1. Shape guard (§3.4). If fail → skip + note.
2. CIE + full layer sweep for the 6 core tasks, seed=42, K and T from the table, `n_mean_activations_trials=50`.
3. From the sweep JSONs: per-task and mean **top-1 ZS FV accuracy** at the model's **peak layer** and at the
   **relative-depth-⅓ layer** (paper's canonical choice ≈ |L|/3); baseline (`clean_topk`) at the same layer.
   Also record SL (shuffled-label) FV accuracy.
4. Aggregate `*_indirect_effect.pt` → **top-K AIE heads + their layer indices**; compute **concentration**
   (fraction of top-K heads in the first ⅔ of layers; mean/median layer as a fraction of depth).
5. `prefix_matching_score` for the top-K heads → induction-head overlap (C13 analog).
6. #heads ablation (C16) for ≤3 tasks at the peak layer.
7. (optional, budget permitting) antonym template portability (C10).

**COST CAP (pre-specified — obey and state in the report):**
- 1 seed (42); 6 core tasks; `n_indirect_effect_trials × n_heads ≈ 400–500` ⇒ T=15 for 28–32-head models
  (25 for GPT-J's 16 heads, already on disk); `n_mean_activations_trials=50`.
- K = round(10 × n_heads/16), capped 20.
- **Hard wall-clock cap 90 min/model.** Over it ⇒ drop to 3-task subset {antonym, country-capital,
  present-past} and record it.
- **20B+ off by default.** NeoX = reuse antonym only. gpt-oss / full-NeoX only with written justification
  (antonym-only, T=10, ≤2 GPU-h). Empirical anchor: NeoX antonym CIE ≈ 2.2 h (on-disk timestamps).
- **Total study ≤ 8 GPU-h.** Estimate: GPT-J/NeoX reuse (0 h) + 4×~45 min ≈ 3 h + overhead ≈ **~5 GPU-h**.
  If OLMo-2 download is dropped, ~4 h. Never exceed 8; cut the last rung and disclose instead.

---

## 5. The over-time figure

Under `followup/001-living-update/results/`, produce **`fv_over_time.pdf`** (matplotlib):
x-axis = model release year (2021 GPT-J, 2022 NeoX[antonym-only], 2023 Llama-2, 2024 Qwen2.5[/Llama-3.1], 2025
Qwen3, OLMo-2); y-axis = **mean top-1 zero-shot FV accuracy** (and baseline no-FV as a faint lower line).
Points labelled with model name and family (color by family: EleutherAI / Llama / Qwen / OLMo), marker shape by
attn type (MHA vs GQA). Overlay each model's baseline→+FV gap. Annotate the GPT-J anchor with the on-disk
55.0 % value. A second small panel (or `fv_peak_layer_vs_depth.pdf`): peak edit-layer as a **fraction of model
depth** vs release year — the input to 002's fitted relation. Keep to ≤3 figures total.

---

## 6. Environment constraints (restate in report)

Saturated SLURM, 1 GPU/study; 7–8B bf16 fits (~13–16 GB weights); ≤8 GPU-h; CIE is the cost (bounded in §4).
Storage under `/net/projects2/chai-lab-models/haokunliu/` only, `HF_HOME=…/alignment-batch/hf-cache`, never
`/home`. Reuse the run's pinned `.venv` (baukit + nnsight + transformers 4.49) — do not upgrade. Gated models
(Llama-3.1 base, Gemma-3) not used as base; ungated fallbacks in the ladder. Prefer BASE models; the single
instruct rung (Llama-3.1-8B-Instruct) is flagged everywhere.

---

## 7. Deliverables (write under `followup/001-living-update/`)

1. **`report.md`** — with a **comparison table** whose columns are: model | year | family | attn | K | mean ZS
   FV % (new, this session) | baseline no-FV % | SL FV % | peak layer (abs / frac depth) | top-AIE
   concentration (frac in first ⅔) | induction overlap | **[original]** GPT-J paper 57.5 & **[artifact-file]**
   replicated 55.0 (cite `results/gptj_seed*/…`). One row per model. Plus: the cost cap actually applied per
   model, any task-subset reductions, any skipped models (with the shape-guard reason), and the precision note
   (GPT-J fp16 on-disk vs new bf16). State GPU-hours used.
2. **`followup_summary.json`** — machine-readable per-model results (accuracies, peak layers, head sets,
   concentration, prefix-match), cost accounting, and the exact commands run.
3. **`result_card.json`** — schema `sai.followup.result_card/v1`:
   - `headline`: one sentence with the key number (e.g. "Zero-shot function-vector accuracy holds across
     2021→2025: mean top-1 rises from GPT-J 55.0% to <best new model> X% while the mid-layer head-localization
     and induction-overlap signatures persist under the MHA→GQA shift").
   - `status`: match / partial / mixed as warranted.
   - 3–5 `metrics`, each with baseline + provenance (baseline = on-disk GPT-J value with its artifact path;
     new value = this session's file).
   - ≤2 tables; 0–3 figures under `results/` (the over-time figure + optional peak-layer-vs-depth).
   - `notes`: every hole, skip, subset, hand-set value, and the base-vs-instruct caveat.
4. **Figures** under `results/`: `fv_over_time.pdf` (required), optional `fv_peak_layer_vs_depth.pdf`,
   optional `fv_head_concentration.pdf`.

Leave `results/` intact for **002-theory-update** to consume (it reuses 001's per-model JSON + head sets, 0 GPU-h).