# Architecture

The encoder matrix maps genes to overlapping biological modules. A gene may contribute to several modules, and unannotated genes are placed in deterministic residual blocks so encoding never silently removes them. The decoder uses ridge regression plus empirical residual resampling to recover gene-level covariance that a module mean alone would erase.

Metacells are built within cell labels. Each stores mean expression, population mass, optional spatial center and local residuals. Forecasting interpolates between bracketing stages when possible and uses a regularized temporal trend beyond the observed range.

Knowledge bundles are versioned tables for modules, signed regulation, associations, cell-type relations, anatomy, phenotypes and expression priors. Evidence from the same family is deduplicated before independent families are combined. Measured expression from protected stages or genotypes is rejected by the leakage audit.

The T3 head estimates the observed Mab21l2 response, gates transfer by module overlap with the target perturbation and can blend it with a signed causal prior. This is a conservative baseline: one observed KO cannot identify a general causal response function without external perturbation data or structured priors.
