# Function Vectors in Large Language Models, a living-paper repository

This repository holds a replication and a set of living follow-up studies built on top of
the paper:

> Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and
> David Bau. **Function Vectors in Large Language Models.** The Twelfth International
> Conference on Learning Representations (ICLR 2024).
> arXiv:2310.15213. OpenReview: https://openreview.net/forum?id=AwyxtyMwaG
> Project page: https://functions.baulab.info

```bibtex
@inproceedings{todd2024function,
    title={Function Vectors in Large Language Models},
    author={Eric Todd and Millicent L. Li and Arnab Sen Sharma and Aaron Mueller and Byron C. Wallace and David Bau},
    booktitle={The Twelfth International Conference on Learning Representations},
    url={https://openreview.net/forum?id=AwyxtyMwaG},
    note={arXiv:2310.15213},
    year={2024},
}
```

ICLR papers are published through OpenReview and do not carry a separate DOI. The stable
identifier for this work is the arXiv id **2310.15213** together with the OpenReview link
above.

## What a living-paper repository is

A living-paper repository starts from a faithful replication of a published paper and then
keeps asking what happens next. The replication reproduces the paper's core results with
open code and documented runs. On top of it sit follow-up studies, each a self-contained
mini-project that revisits a claim under new conditions (newer models, a fresh analysis, a
different architecture) and records what it found. The idea is that the paper keeps
breathing: as the field moves, the studies accumulate and the picture is updated rather
than frozen at publication time.

Everything here is organized so you can trace every reported number back to a file that a
real run produced. Model weights and large raw activation tensors are not committed (they
are large and can be regenerated); see `LARGE_FILES_OMITTED.md`.

## What this repo contains

### Replication (`run/replication/`)

A GPT-J-6B replication of the paper's core claims, head localization by causal indirect
effect, zero-shot and shuffled-label causal task triggering, per-task function-vector
accuracy, and the layer-dependence sweep, with the intervention machinery implemented in
`nnsight` (as directed) while reusing the official repository's prompt, dataset, seed, and
evaluation logic. Mean zero-shot function-vector accuracy replicated at 55.0% against the
paper's 57.5% (six core tasks, edit layer 9), with GPT-NeoX-20B as the ungated cross-model
check. The gated Llama-2 rows were blocked by access limits at replication time and are
noted as an environmental limitation. The working copy of the official code lives under
`run/replication/codebase/`; the run logs, `replication_log.json`, and `codebase.diff`
document exactly what was run and changed.

### Follow-up studies (`run/followup/`)

- **001-living-update** — Do function vectors survive in today's models? Zero-shot
  function vectors survive 2021 to 2025 and the MHA to GQA shift: mean top-1 accuracy at
  the peak edit layer rises from GPT-J 65.9% to Llama-2 92.2% and OLMo-2 89.3%, with the
  mid-layer head-localization signature intact, but the optimal edit layer drifts later in
  the Qwen (GQA) family, breaking the paper's depth-over-three heuristic.

- **002-theory-update** — How function-vector behavior scales with era, family, and
  architecture (analysis only, no new GPU work). The trigger tracks about one third of
  network depth for MHA models (peak layer near 0.30 times depth across three families and
  2021 to 2025, confirming the paper's L/3), but the MHA to GQA shift moves the trigger to
  about 0.6 times depth in the Qwen family, so a single depth rule collapses across the
  full ladder; the effect size itself holds high (45 to 92%) with no monotone era trend.

- **003-living-update** — Extending the ladder to 2026 models. The zero-shot
  function-vector effect is still detectable but much weaker on 2026 multimodal-architecture
  models: mean top-1 accuracy at the peak edit layer falls from Llama-2 92.2% and OLMo-2
  89.3% (2023 to 2025) to Qwen3.5-9B 16.9% and gemma-4-12B antonym 18.0%. The function
  vector still beats the no-FV baseline in every completed task, so the causal effect is
  real but small, and the peak edit layer stays mid-to-late. A methodological caveat is
  recorded: the 2026 head decomposition used transformers forward hooks rather than
  nnsight, so part of the attenuation may be measurement rather than a pure model property.

Each study folder carries its own `instruction.md`, `report.md`, `result_card.json`,
`followup_summary.json`, a `workspace/` with the scripts and logs that produced it, and a
`results/` with the figures and JSON deliverables.

## Living page

The living page for this paper is at
https://livingscience.ai/safety/living-function-vectors

## Reuse and authors' terms

The replicated code, the demo notebook, and the task datasets come from the official
`function_vectors` release by the paper's authors
(https://github.com/ericwtodd/function_vectors) and remain under their original license,
preserved at `run/replication/codebase/LICENSE`. Please honor the original authors' terms
when you reuse that material, and cite the paper above. The follow-up studies and the
replication driver code added on top are released in the same spirit for research use.

## Large files

Model weights and large raw activation and causal-indirect-effect tensors are not committed
to keep the repository lightweight. Every omission, what it is, and how to regenerate it is
listed in `LARGE_FILES_OMITTED.md`.
