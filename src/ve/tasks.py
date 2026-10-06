from __future__ import annotations

import numpy as np

from .core import (GeneModules, ModuleDecoder, build_metacells, communication_step,
                   dense, forecast, perturbation_modules, sample_state)
from .knowledge import KnowledgeBundle


def _ct(ad):
    return ad.obs["celltype"].astype(str).to_numpy() if "celltype" in ad.obs else None


def _fit_decoder(matrices, encoded, seed):
    rng = np.random.default_rng(seed); xs, zs = [], []
    per_stage = max(256, 4096 // len(matrices))
    for x, z in zip(matrices, encoded):
        take = rng.choice(x.shape[0], min(per_stage, x.shape[0]), replace=False)
        xs.append(dense(x[take])); zs.append(z[take])
    return ModuleDecoder.fit(np.vstack(zs), np.vstack(xs), max_residuals=4096, seed=seed)


def _modules(genes, go_path=None, knowledge_path=None, task=None):
    if knowledge_path:
        kb = KnowledgeBundle.load(knowledge_path, task)
        return GeneModules.from_knowledge(genes, kb), kb
    if not go_path: raise ValueError("either --knowledge or --go is required")
    return GeneModules.from_tsv(genes, go_path), None


def predict_temporal(adatas, times, target_time, go_path=None, n_cells=None, metacell_size=64, seed=0,
                     knowledge_path=None):
    genes = list(map(str, adatas[0].var_names))
    if any(list(map(str, a.var_names)) != genes for a in adatas): raise ValueError("input gene orders differ")
    gm, kb = _modules(genes, go_path, knowledge_path, "task1")
    matrices = [a.X for a in adatas]; encoded = [gm.encode(x) for x in matrices]
    decoder = _fit_decoder(matrices, encoded, seed)
    states = [build_metacells(z, _ct(a), None, metacell_size, seed + i) for i, (a, z) in enumerate(zip(adatas, encoded))]
    pred = forecast(states, list(times), target_time, kb, gm.modules)
    rng = np.random.default_rng(seed)
    z, _ = sample_state(pred, int(n_cells or adatas[np.argmax(times)].n_obs), rng)
    return decoder.decode(z, rng), genes, gm.modules


def predict_spatial(adatas, times, target_time, go_path=None, n_cells=None, metacell_size=64, seed=0,
                    knowledge_path=None, communication_strength=.05):
    genes = list(map(str, adatas[0].var_names)); gm, kb = _modules(genes, go_path, knowledge_path, "task2")
    if any(list(map(str, a.var_names)) != genes for a in adatas): raise ValueError("input gene orders differ")
    if any("spatial_3D" not in a.obsm for a in adatas): raise ValueError("Task 2 requires obsm['spatial_3D']")
    matrices = [a.X for a in adatas]; encoded = [gm.encode(x) for x in matrices]
    decoder = _fit_decoder(matrices, encoded, seed)
    states = [build_metacells(z, _ct(a), a.obsm["spatial_3D"], metacell_size, seed + i)
              for i, (a, z) in enumerate(zip(adatas, encoded))]
    pred = forecast(states, list(times), target_time, kb, gm.modules)
    if kb is not None: pred = communication_step(pred, gm.modules, communication_strength)
    rng = np.random.default_rng(seed)
    z, xyz = sample_state(pred, int(n_cells or adatas[np.argmax(times)].n_obs), rng)
    return decoder.decode(z, rng), xyz.astype(np.float32), genes, gm.modules


def predict_perturbation(wt_train, ko_train, wt_target, source_gene, target_gene, go_path=None,
                         n_cells=None, metacell_size=64, seed=0, knowledge_path=None,
                         causal_strength=.65, communication_strength=.03):
    genes = list(map(str, wt_target.var_names)); gm, kb = _modules(genes, go_path, knowledge_path, "task3")
    for a in (wt_train, ko_train):
        if list(map(str, a.var_names)) != genes: raise ValueError("input gene orders differ")
    matrices = [wt_train.X, ko_train.X, wt_target.X]
    zwt, zko, ztarget = [gm.encode(x) for x in matrices]
    decoder = _fit_decoder(matrices, [zwt, zko, ztarget], seed)
    delta = zko.mean(0) - zwt.mean(0)
    gate = perturbation_modules(gm, source_gene, target_gene)
    state = build_metacells(ztarget, _ct(wt_target), wt_target.obsm.get("spatial_3D"), metacell_size, seed)
    empirical = delta * gate
    if kb is not None:
        causal = kb.module_causal_effect(genes, gm.weights, target_gene)
        scale = np.quantile(np.abs(delta), .9) if len(delta) else 0.
        signed = causal * max(float(scale), 1e-3)
        phenotype = kb.phenotype_gate(target_gene)
        effect = (1 - causal_strength) * empirical + causal_strength * signed * (.5 + .5 * phenotype)
    else:
        effect = empirical
    # Mesp1-Cre perturbations are mesoderm restricted; activity in each metacell gates the response.
    target_idx = genes.index(target_gene) if target_gene in genes else None
    if target_idx is not None:
        target_modules = gm.weights[target_idx]
        activity = state.expression @ target_modules
        activity /= max(float(np.quantile(activity, .9)), 1e-6)
        activity = np.clip(activity, 0, 1)
    else: activity = np.ones(len(state.expression), dtype=np.float32)
    mesoderm = np.asarray([1. if any(x in str(label).lower() for x in ("mesoderm", "cardiac", "heart", "endothel", "muscle")) else .15
                           for label in state.labels], dtype=np.float32)
    state.expression = np.maximum(state.expression + activity[:, None] * mesoderm[:, None] * effect[None, :], 0)
    if kb is not None: state = communication_step(state, gm.modules, communication_strength)
    # Geometry is retained from matched WT: current Task-3 ranking has no spatial shape term.
    rng = np.random.default_rng(seed)
    z, xyz = sample_state(state, int(n_cells or wt_target.n_obs), rng)
    return decoder.decode(z, rng), xyz.astype(np.float32), genes, gm.modules
