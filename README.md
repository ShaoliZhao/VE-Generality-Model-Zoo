# VEC Generality Model Zoo

One runnable modeling system and controlled baseline zoo for all three Virtual Embryo tasks, built around a shared metacell and overlapping biological-module state space.

Most task baselines become three unrelated scripts. This repository instead tests a scientific hypothesis: temporal development, morphogenesis and perturbation can share a population representation while using task-specific transition operators.

## Shared state

1. Cells are grouped into metacells within annotated cell states.
2. Genes are encoded into overlapping modules; unannotated genes are retained in residual blocks.
3. A decoder reconstructs cell-level gene expression and resamples residual variation.
4. Every stochastic operation is seeded and every run writes hashes and parameters.

## Task heads

- **T1:** low-capacity temporal forecasting of metacell composition and module expression, followed by gene-level decoding.
- **T2:** interpolation or extrapolation of expression, population proportions, normalized geometry and physical scale; optional spatial communication update.
- **T3:** matched-WT state plus a signed, context-gated perturbation delta. Mesp1-lineage gating prevents a conditional KO from being applied uniformly to unrelated cells.

## Install and run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
ve --help
```

Example commands and required AnnData fields are documented in [task recipes](docs/task-recipes.md). `knowledge/example` is a schema fixture, not a claimed biological knowledge release. Replace it with a versioned, audited bundle before scientific use.

## Why a unified repository

Shared code makes task transfer testable. An ablation can compare task-specific models with shared modules, shared decoders, shared cell-state priors or the complete system. It also makes negative transfer visible instead of hiding it behind different preprocessing pipelines.

The implementation never claims that one latent trajectory solves all three tasks. T1 is long-horizon extrapolation, T2 includes unregistered geometry, and T3 is unseen-intervention generalization. Task-specific heads remain separate because those observation models differ.

See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md), [literature](references/reading.md), and the [Community Award draft](SUBMISSION_TEXT.md).

MIT licensed. No challenge data are included.
