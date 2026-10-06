from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
from typing import Iterable

import numpy as np

from .knowledge import KnowledgeBundle


def dense(x) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


@dataclass
class GeneModules:
    genes: list[str]
    modules: list[str]
    weights: np.ndarray  # gene × module, each gene sums to one

    @classmethod
    def from_tsv(cls, genes: Iterable[str], path: str | Path, unannotated_chunk: int = 128):
        genes = [str(g) for g in genes]
        index = {g: i for i, g in enumerate(genes)}
        rows: list[tuple[int, str, float]] = []
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or not {"gene", "module"}.issubset(reader.fieldnames):
                raise ValueError("GO TSV must contain gene and module columns")
            for row in reader:
                if row["gene"] in index:
                    rows.append((index[row["gene"]], row["module"], float(row.get("weight") or 1.0)))
        annotated = {i for i, _, _ in rows}
        for k, start in enumerate(range(0, len(genes), unannotated_chunk)):
            name = f"UNANNOTATED_{k:04d}"
            rows.extend((i, name, 1.0) for i in range(start, min(start + unannotated_chunk, len(genes))) if i not in annotated)
        modules = sorted({m for _, m, _ in rows})
        if not modules:
            raise ValueError("No input genes were found in GO TSV")
        midx = {m: j for j, m in enumerate(modules)}
        w = np.zeros((len(genes), len(modules)), dtype=np.float32)
        for i, m, value in rows:
            w[i, midx[m]] += max(value, 0.0)
        sums = w.sum(1, keepdims=True)
        missing = np.flatnonzero(sums[:, 0] == 0)
        if len(missing):
            raise AssertionError("internal error: genes were dropped")
        w /= sums
        return cls(genes, modules, w)

    @classmethod
    def from_knowledge(cls, genes: Iterable[str], knowledge: KnowledgeBundle, unannotated_chunk: int = 128):
        genes = [str(g) for g in genes]; index = {g: i for i, g in enumerate(genes)}
        rows = [(index[r["gene"]], r["module"], float(r.get("weight") or 1))
                for r in knowledge.module_rows() if r.get("gene") in index]
        annotated = {i for i, _, _ in rows}
        for k, start in enumerate(range(0, len(genes), unannotated_chunk)):
            rows.extend((i, f"UNANNOTATED_{k:04d}", 1.) for i in range(start, min(start + unannotated_chunk, len(genes))) if i not in annotated)
        modules = sorted({m for _, m, _ in rows}); midx = {m: j for j, m in enumerate(modules)}
        w = np.zeros((len(genes), len(modules)), dtype=np.float32)
        for i, m, value in rows: w[i, midx[m]] += max(value, 0.)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-8)
        return cls(genes, modules, w)

    def encode(self, x: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.weights.sum(0), 1e-8)
        out = np.empty((x.shape[0], len(self.modules)), dtype=np.float32)
        for start in range(0, x.shape[0], 4096):
            block = x[start:start + 4096]
            projected = block @ self.weights
            out[start:start + len(block)] = dense(projected) / denom
        return out


@dataclass
class ModuleDecoder:
    coef: np.ndarray
    residuals: np.ndarray

    @classmethod
    def fit(cls, z: np.ndarray, x: np.ndarray, max_residuals: int = 4096, seed: int = 0):
        rng = np.random.default_rng(seed)
        take = rng.choice(x.shape[0], min(max_residuals, x.shape[0]), replace=False)
        z = dense(z[take]); x = dense(x[take])
        design = np.column_stack([np.ones(len(z), dtype=np.float32), z])
        coef = np.linalg.lstsq(design, x, rcond=1e-5)[0].astype(np.float32)
        residuals = (x - design @ coef).astype(np.float32)
        return cls(coef, residuals)

    def decode(self, z: np.ndarray, rng: np.random.Generator, noise: float = 1.0):
        design = np.column_stack([np.ones(len(z), dtype=np.float32), z])
        out = design @ self.coef
        if len(self.residuals) and noise:
            out += noise * self.residuals[rng.integers(0, len(self.residuals), len(out))]
        return np.maximum(out, 0).astype(np.float32)


@dataclass
class MetaState:
    expression: np.ndarray
    spatial: np.ndarray | None
    weights: np.ndarray
    labels: np.ndarray
    expr_residuals: list[np.ndarray]
    spatial_residuals: list[np.ndarray] | None


def _balanced_metacells(x: np.ndarray, target_size: int, rng: np.random.Generator) -> np.ndarray:
    """Scalable balanced partition along a data-derived state-space ordering."""
    if len(x) <= target_size:
        return np.zeros(len(x), dtype=np.int32)
    sample = x[rng.choice(len(x), min(4096, len(x)), replace=False)]
    mean = sample.mean(0); direction = rng.normal(size=x.shape[1]).astype(np.float32)
    # A few power iterations approximate the leading state axis without materialising n×k distances.
    for _ in range(5):
        direction = (sample - mean).T @ ((sample - mean) @ direction)
        direction /= max(np.linalg.norm(direction), 1e-8)
    score = (x - mean) @ direction
    order = np.argsort(score, kind="stable")
    labels = np.empty(len(x), dtype=np.int32)
    labels[order] = np.arange(len(x), dtype=np.int32) // target_size
    return labels


def build_metacells(z: np.ndarray, celltypes: Iterable[str] | None, spatial: np.ndarray | None,
                    target_size: int, seed: int) -> MetaState:
    z = dense(z)
    spatial = None if spatial is None else dense(spatial)[:, :3]
    types = np.asarray(list(celltypes) if celltypes is not None else ["all"] * len(z), dtype=str)
    rng = np.random.default_rng(seed)
    assignments = np.empty(len(z), dtype=np.int32)
    offset = 0
    for ct in sorted(set(types)):
        idx = np.flatnonzero(types == ct)
        features = z[idx]
        if spatial is not None:
            s = spatial[idx]
            s = (s - s.mean(0)) / np.maximum(s.std(0), 1e-6)
            features = np.column_stack([features, s])
        local = _balanced_metacells(features, target_size, rng)
        assignments[idx] = local + offset
        offset += int(local.max()) + 1
    means, coords, weights, labels, er, sr = [], [], [], [], [], []
    for j in range(offset):
        idx = np.flatnonzero(assignments == j)
        mu = z[idx].mean(0)
        means.append(mu); weights.append(len(idx)); labels.append(types[idx[0]])
        keep = idx if len(idx) <= 32 else rng.choice(idx, 32, replace=False)
        er.append((z[keep] - mu).astype(np.float32))
        if spatial is not None:
            smu = spatial[idx].mean(0); coords.append(smu); sr.append((spatial[keep] - smu).astype(np.float32))
    return MetaState(np.asarray(means), np.asarray(coords) if spatial is not None else None,
                     np.asarray(weights, dtype=np.float32), np.asarray(labels), er, sr if spatial is not None else None)


def _nearest(source: np.ndarray, query: np.ndarray, source_labels=None, query_labels=None,
             knowledge: KnowledgeBundle | None = None, lineage_strength: float = 0.25) -> np.ndarray:
    out = np.empty(len(query), dtype=np.int32)
    source_norm = (source ** 2).sum(1)[None, :]
    for start in range(0, len(query), 512):
        q = query[start:start + 512]
        distance = (q ** 2).sum(1)[:, None] + source_norm - 2 * (q @ source.T)
        if knowledge is not None and source_labels is not None and query_labels is not None:
            scale = max(float(np.median(distance)), 1e-6) * lineage_strength
            penalty = np.asarray([[knowledge.cell_distance(str(s), str(t)) for s in source_labels]
                                  for t in query_labels[start:start + len(q)]], dtype=np.float32)
            distance += scale * penalty
        out[start:start + len(q)] = distance.argmin(1)
    return out


def forecast(states: list[MetaState], times: list[float], target: float,
             knowledge: KnowledgeBundle | None = None, module_names: list[str] | None = None) -> MetaState:
    order = np.argsort(times); states = [states[i] for i in order]; times = [times[i] for i in order]
    if len(states) < 2:
        return states[-1]
    if target <= times[0]:
        left, right = 0, 1
    elif target >= times[-1]:
        left, right = len(times) - 2, len(times) - 1
    else:
        right = int(np.searchsorted(times, target, side="right")); left = right - 1
    a, b = states[left], states[right]
    alpha = (target - times[right]) / (times[right] - times[left])
    match = _nearest(a.expression, b.expression, a.labels, b.labels, knowledge)
    expr = np.maximum(b.expression + alpha * (b.expression - a.expression[match]), 0)
    weights = np.maximum(1, b.weights + alpha * (b.weights - a.weights[match]))
    if knowledge is not None:
        compatibility = np.asarray([knowledge.anatomy_compatibility(str(x), target) for x in b.labels])
        weights *= .5 + compatibility
    spatial = None
    if b.spatial is not None and a.spatial is not None:
        # Center each specimen: evaluation is translation/rotation invariant.
        ac = a.spatial - np.average(a.spatial, axis=0, weights=a.weights)
        bc = b.spatial - np.average(b.spatial, axis=0, weights=b.weights)
        scale_a = np.sqrt(np.average((ac ** 2).sum(1), weights=a.weights))
        scale_b = np.sqrt(np.average((bc ** 2).sum(1), weights=b.weights))
        scale = max(1e-6, scale_b + alpha * (scale_b - scale_a))
        spatial = bc * (scale / max(scale_b, 1e-6))
    labels = b.labels.copy(); er = list(b.expr_residuals); sr = None if b.spatial_residuals is None else list(b.spatial_residuals)
    if knowledge is not None and target > times[right]:
        extra_expr, extra_weights, extra_labels, extra_er, extra_xyz, extra_sr = [], [], [], [], [], []
        module_index = {m: i for i, m in enumerate(module_names or [])}
        for i, label in enumerate(labels):
            for child, rate, marker in knowledge.lineage_births(str(label)):
                child_expr = expr[i].copy()
                if marker in module_index: child_expr[module_index[marker]] *= 1.25
                extra_expr.append(child_expr); extra_weights.append(max(1, weights[i] * rate * (target - times[right])))
                extra_labels.append(child); extra_er.append(er[i])
                if spatial is not None: extra_xyz.append(spatial[i]); extra_sr.append(sr[i])
        if extra_expr:
            expr = np.vstack([expr, extra_expr]); weights = np.r_[weights, extra_weights]; labels = np.r_[labels, extra_labels]
            er.extend(extra_er)
            if spatial is not None: spatial = np.vstack([spatial, extra_xyz]); sr.extend(extra_sr)
    return MetaState(expr, spatial, weights, labels, er, sr)


def communication_step(state: MetaState, module_names: list[str], strength: float = 0.05,
                       neighbors: int = 16) -> MetaState:
    """Spatial ligand→receptor coupling on metacells, gated by co-expression and proximity."""
    if state.spatial is None or strength <= 0: return state
    idx = {m: i for i, m in enumerate(module_names)}
    pathways = sorted({m.split("::", 1)[1] for m in module_names if m.startswith("LIGAND::")
                       and "RECEPTOR::" + m.split("::", 1)[1] in idx})
    if not pathways: return state
    xyz = state.spatial - state.spatial.mean(0); norm = (xyz ** 2).sum(1)[None, :]
    k = min(max(1, neighbors), max(1, len(xyz) - 1))
    neighbor_ids = np.empty((len(xyz), k), dtype=np.int32)
    neighbor_dist = np.empty((len(xyz), k), dtype=np.float32)
    for start in range(0, len(xyz), 512):
        q = xyz[start:start + 512]
        d2 = np.maximum((q ** 2).sum(1)[:, None] + norm - 2 * q @ xyz.T, 0)
        rows = np.arange(len(q)); d2[rows, start + rows] = np.inf
        ids = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        neighbor_ids[start:start + len(q)] = ids
        neighbor_dist[start:start + len(q)] = np.take_along_axis(d2, ids, axis=1)
    bandwidth = max(float(np.median(neighbor_dist[np.isfinite(neighbor_dist)])), 1e-6)
    affinity = np.exp(-neighbor_dist / bandwidth)
    affinity /= np.maximum(affinity.sum(1, keepdims=True), 1e-8)
    expression = state.expression.copy()
    for pathway in pathways:
        li, ri = idx[f"LIGAND::{pathway}"], idx[f"RECEPTOR::{pathway}"]
        incoming = (affinity * expression[neighbor_ids, li]).sum(1)
        signal = incoming * expression[:, ri]
        signal /= max(float(np.quantile(signal, .95)), 1e-6)
        expression[:, ri] *= 1 + strength * np.clip(signal, 0, 2)
        downstream = idx.get(f"SIGNAL::{pathway}")
        if downstream is not None: expression[:, downstream] += strength * signal
    state.expression = np.maximum(expression, 0)
    return state


def sample_state(state: MetaState, n_cells: int, rng: np.random.Generator):
    p = state.weights / state.weights.sum()
    ids = rng.choice(len(p), n_cells, p=p)
    z = state.expression[ids].copy()
    xyz = None if state.spatial is None else state.spatial[ids].copy()
    for row, m in enumerate(ids):
        r = state.expr_residuals[m]
        if len(r): z[row] += r[rng.integers(len(r))]
        if xyz is not None and state.spatial_residuals is not None:
            r3 = state.spatial_residuals[m]
            if len(r3): xyz[row] += r3[rng.integers(len(r3))]
    return np.maximum(z, 0), xyz


def perturbation_modules(modules: GeneModules, source_gene: str, target_gene: str) -> np.ndarray:
    idx = {g: i for i, g in enumerate(modules.genes)}
    if target_gene not in idx:
        raise ValueError(f"target perturbation gene {target_gene!r} is absent from the panel")
    target = modules.weights[idx[target_gene]]
    if source_gene in idx:
        source = modules.weights[idx[source_gene]]
        # Shared annotation gets full transfer; target-only biological context gets a conservative half transfer.
        return np.clip(source * target + 0.5 * target, 0, 1)
    return target


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_run_record(output: str | Path, command: str, inputs: list[str], parameters: dict, summary: dict):
    record = {"command": command, "inputs": [{"path": p, "sha256": sha256(p)} for p in inputs],
              "parameters": parameters, "summary": summary}
    Path(str(output) + ".run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
