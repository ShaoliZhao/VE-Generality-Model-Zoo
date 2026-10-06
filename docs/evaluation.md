# Evaluation and ablations

Use whole-stage pseudo-holdouts. Random cell splits leak the population distribution and do not test forecasting.

Compare: copy-last/WT identity; gene-wise linear trend; metacells without modules; modules without residual resampling; shared representation with separate heads; full model. For T2 separately ablate geometry normalization, physical scale and communication. For T3 ablate empirical KO transfer, signed prior, lineage gate and phenotype gate.

Select hyperparameters on released pseudo-holdouts before leaderboard submission. Record the official scorer version, panel hash, configuration, seed, input hashes and output hash. Negative transfer is a result: if sharing a decoder or module state worsens a task, report it rather than tuning separate preprocessing invisibly.
