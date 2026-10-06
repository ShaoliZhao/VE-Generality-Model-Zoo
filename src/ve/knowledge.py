from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from collections import defaultdict, deque

import numpy as np


FORBIDDEN = {
    "task1": {"stages": {"E10.5", "E12.5"}, "genotypes": set()},
    "task2": {"stages": {"E7.5", "E7.75", "E8.5", "E10.5", "E12.5"}, "genotypes": set()},
    "task3": {"stages": set(), "genotypes": {"Gata4_KO", "beta-catenin_KO", "Ctnnb1_KO"}},
}


def _rows(path: Path):
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


@dataclass
class KnowledgeBundle:
    root: Path
    manifest: dict
    modules: list[dict]
    regulatory: list[dict]
    interactions: list[dict]
    cell_types: list[dict]
    expression_priors: list[dict]
    associations: list[dict]
    phenotypes: list[dict]
    anatomy: list[dict]

    @classmethod
    def load(cls, root: str | Path, task: str | None = None):
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists(): raise ValueError(f"knowledge manifest missing: {manifest_path}")
        obj = cls(root, json.loads(manifest_path.read_text()), _rows(root / "modules.tsv"),
                  _rows(root / "regulatory.tsv"), _rows(root / "interactions.tsv"),
                  _rows(root / "cell_types.tsv"), _rows(root / "expression_priors.tsv"),
                  _rows(root / "associations.tsv"), _rows(root / "phenotypes.tsv"),
                  _rows(root / "anatomy.tsv"))
        obj.validate(task)
        return obj

    def validate(self, task: str | None = None):
        if not self.manifest.get("resources"): raise ValueError("manifest.resources must not be empty")
        required = {"name", "version", "url", "license", "species", "kind"}
        for resource in self.manifest["resources"]:
            missing = required - set(resource)
            if missing: raise ValueError(f"resource lacks {sorted(missing)}: {resource}")
            if task and resource.get("measured_expression", False):
                bad = FORBIDDEN[task]
                stages = set(resource.get("developmental_stages", []))
                genotypes = set(resource.get("genotypes", []))
                if stages & bad["stages"] or genotypes & bad["genotypes"]:
                    raise ValueError(f"held-out leakage in resource {resource['name']}: stages={stages & bad['stages']}, genotypes={genotypes & bad['genotypes']}")
        for row in self.regulatory:
            if row.get("sign", "0") not in {"-1", "0", "1", "-1.0", "0.0", "1.0"}:
                raise ValueError("regulatory.sign must be -1, 0, or 1")

    @staticmethod
    def _reliability(row):
        confidence = float(row.get("confidence") or 0.5)
        species = float(row.get("species_match") or 1.0)
        context = float(row.get("context_match") or 0.5)
        directness = float(row.get("directness") or 0.5)
        return float(np.clip(confidence * species * (.5 + .5 * context) * (.5 + .5 * directness), 0, 1))

    def association_consensus(self, genes: list[str]):
        """Evidence-family-aware gene network; correlated databases do not receive duplicate votes."""
        idx = {g: i for i, g in enumerate(genes)}; evidence = defaultdict(dict)
        for row in self.associations:
            a, b = row.get("gene_a"), row.get("gene_b")
            if a not in idx or b not in idx or a == b: continue
            key = tuple(sorted((a, b))); family = row.get("evidence_family") or row.get("source") or "unknown"
            evidence[key][family] = max(evidence[key].get(family, 0), self._reliability(row))
        network = np.zeros((len(genes), len(genes)), dtype=np.float32)
        for (a, b), families in evidence.items():
            # Noisy-OR combines independent evidence families after max-deduplicating each family.
            score = 1 - np.prod([1 - v for v in families.values()])
            network[idx[a], idx[b]] = network[idx[b], idx[a]] = score
        return network

    def diffuse_gene_effect(self, genes: list[str], seed_effect: np.ndarray, restart: float = .7, steps: int = 5):
        network = self.association_consensus(genes)
        if not np.any(network): return seed_effect.astype(np.float32)
        transition = network / np.maximum(network.sum(1, keepdims=True), 1e-8)
        value = seed_effect.astype(np.float32).copy()
        for _ in range(steps): value = restart * seed_effect + (1 - restart) * (transition.T @ value)
        return np.clip(value, -1, 1)

    def phenotype_gate(self, gene: str, keywords=("heart", "cardiac", "embry", "vascul", "morphogenesis")):
        rows = [r for r in self.phenotypes if r.get("gene") == gene]
        if not rows: return 0.5
        relevant = [self._reliability(r) for r in rows if any(k in (r.get("phenotype", "") + " " + r.get("anatomy", "")).lower() for k in keywords)]
        return float(np.clip(max(relevant, default=.15), .15, 1.))

    def anatomy_compatibility(self, cell_type: str, stage: float | None = None):
        rows = [r for r in self.anatomy if r.get("cell_type") == cell_type]
        if not rows: return 0.5
        scores = []
        for row in rows:
            lo = float(row.get("stage_start") or -np.inf); hi = float(row.get("stage_end") or np.inf)
            if stage is None or lo <= stage <= hi: scores.append(self._reliability(row))
        return max(scores, default=.1)

    def module_rows(self):
        rows = list(self.modules)
        for e in self.regulatory:
            confidence = float(e.get("confidence") or 1)
            rows.append({"gene": e["target"], "module": f"REGULON::{e['regulator']}", "weight": confidence})
        for e in self.interactions:
            confidence = float(e.get("confidence") or 1); pathway = e.get("pathway") or "unknown"
            rows.extend([
                {"gene": e["ligand"], "module": f"LIGAND::{pathway}", "weight": confidence},
                {"gene": e["receptor"], "module": f"RECEPTOR::{pathway}", "weight": confidence},
            ])
        for e in self.expression_priors:
            rows.append({"gene": e["gene"], "module": f"CONTEXT::{e['context']}", "weight": float(e.get("confidence") or 0.25)})
        for e in self.associations:
            if e.get("relation") in {"pathway", "complex", "domain", "colocalization"}:
                name = e.get("group") or e.get("relation")
                weight = self._reliability(e)
                rows.extend([{"gene": e["gene_a"], "module": f"NETWORK::{name}", "weight": weight},
                             {"gene": e["gene_b"], "module": f"NETWORK::{name}", "weight": weight}])
        return rows

    def causal_gene_effect(self, genes: list[str], perturbed_gene: str, depth: int = 3, decay: float = .55):
        """Signed loss-of-function propagation through TF/signalling edges."""
        idx = {g: i for i, g in enumerate(genes)}; effect = np.zeros(len(genes), dtype=np.float32)
        graph = defaultdict(list)
        for e in self.regulatory:
            sign = float(e.get("sign") or 0); conf = float(e.get("confidence") or 1)
            if sign: graph[e["regulator"]].append((e["target"], sign, conf))
        # KO is a negative intervention; activation targets go down, repression targets go up.
        q = deque([(perturbed_gene, -1.0, 0)]); best = {perturbed_gene: 1.0}
        while q:
            gene, value, d = q.popleft()
            if gene in idx: effect[idx[gene]] += value
            if d == depth: continue
            for target, sign, conf in graph.get(gene, []):
                nxt = value * sign * conf * decay
                if abs(nxt) > best.get(target, 0): best[target] = abs(nxt); q.append((target, nxt, d + 1))
        effect = np.clip(effect, -1, 1)
        diffused = self.diffuse_gene_effect(genes, effect)
        gate = self.phenotype_gate(perturbed_gene)
        return np.clip((.75 * effect + .25 * diffused) * (.5 + .5 * gate), -1, 1)

    def module_causal_effect(self, genes: list[str], weights: np.ndarray, perturbed_gene: str):
        gene_effect = self.causal_gene_effect(genes, perturbed_gene)
        denom = np.maximum(np.abs(gene_effect) @ weights, 1e-8)
        raw = gene_effect @ weights / denom
        return np.clip(raw, -1, 1).astype(np.float32)

    def cell_distance(self, source: str, target: str) -> float:
        parent = {r["cell_type"]: r.get("parent", "") for r in self.cell_types}
        def ancestors(x):
            out = {x: 0}; d = 0
            while x in parent and parent[x] and parent[x] not in out:
                x = parent[x]; d += 1; out[x] = d
            return out
        a, b = ancestors(source), ancestors(target); common = set(a) & set(b)
        return float(min((a[x] + b[x] for x in common), default=8))

    def lineage_births(self, parent_label: str):
        out = []
        for row in self.cell_types:
            if row.get("parent") == parent_label and float(row.get("birth_rate") or 0) > 0:
                out.append((row["cell_type"], float(row["birth_rate"]), row.get("marker_module", "")))
        return out


def compile_bundle(output: str | Path, sources: list[str | Path], manifest: str | Path):
    """Normalize user-downloaded, version-pinned TSV resources into one bundle."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(manifest).read_text()); (output / "manifest.json").write_text(json.dumps(spec, indent=2) + "\n")
    schemas = {
        "modules": ["gene", "module", "weight", "source"],
        "regulatory": ["regulator", "target", "sign", "confidence", "source"],
        "interactions": ["ligand", "receptor", "pathway", "confidence", "source"],
        "cell_types": ["cell_type", "ontology_id", "parent", "birth_rate", "marker_module", "source"],
        "expression_priors": ["gene", "context", "confidence", "source"],
        "associations": ["gene_a", "gene_b", "relation", "group", "confidence", "evidence_family", "species_match", "context_match", "directness", "source"],
        "phenotypes": ["gene", "phenotype_id", "phenotype", "anatomy", "allele_type", "confidence", "evidence_family", "species_match", "context_match", "directness", "source"],
        "anatomy": ["cell_type", "anatomy_id", "anatomy", "stage_start", "stage_end", "confidence", "species_match", "context_match", "directness", "source"],
    }
    collected = {k: [] for k in schemas}
    for source in map(Path, sources):
        kind = source.stem.split(".")[0]
        if kind not in schemas: raise ValueError(f"unknown knowledge table kind {kind!r} in {source.name}")
        collected[kind].extend(_rows(source))
    for kind, fields in schemas.items():
        with (output / f"{kind}.tsv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            seen = set()
            for row in collected[kind]:
                key = tuple(row.get(x, "") for x in fields)
                if key not in seen: writer.writerow({x: row.get(x, "") for x in fields}); seen.add(key)
    return KnowledgeBundle.load(output)
