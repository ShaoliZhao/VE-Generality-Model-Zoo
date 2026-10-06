from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

from .core import write_run_record
from .knowledge import KnowledgeBundle, compile_bundle
from .tasks import predict_perturbation, predict_spatial, predict_temporal


def _anndata():
    try:
        import anndata as ad
        return ad
    except ImportError as e:
        raise SystemExit("anndata is required; run: pip install -e .") from e


def _write(output, x, genes, spatial=None, metadata=None):
    ad = _anndata(); out = ad.AnnData(X=np.asarray(x, dtype=np.float32))
    out.var_names = genes; out.obs_names = [f"pred_{i:08d}" for i in range(out.n_obs)]
    if spatial is not None: out.obsm["spatial_3D"] = np.asarray(spatial, dtype=np.float32)
    for k, v in (metadata or {}).items(): out.uns[k] = v
    Path(output).parent.mkdir(parents=True, exist_ok=True); out.write_h5ad(output)


def _shared(p):
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--knowledge"); group.add_argument("--go")
    p.add_argument("--output", required=True)
    p.add_argument("--cells", type=int); p.add_argument("--metacell-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)


def parser():
    p = argparse.ArgumentParser(prog="ve"); sub = p.add_subparsers(dest="command", required=True)
    for name in ("task1", "task2"):
        q = sub.add_parser(name); q.add_argument("--inputs", nargs="+", required=True)
        q.add_argument("--times", nargs="+", required=True, type=float); q.add_argument("--target-time", type=float, required=True); _shared(q)
    q = sub.add_parser("task3"); q.add_argument("--wt-train", required=True); q.add_argument("--ko-train", required=True)
    q.add_argument("--wt-target", required=True); q.add_argument("--source-gene", required=True); q.add_argument("--target-gene", required=True); _shared(q)
    q.add_argument("--causal-strength", type=float, default=.65); q.add_argument("--communication-strength", type=float, default=.03)
    q = sub.add_parser("compile-knowledge"); q.add_argument("--manifest", required=True); q.add_argument("--sources", nargs="+", required=True); q.add_argument("--output", required=True)
    q = sub.add_parser("audit-knowledge"); q.add_argument("--knowledge", required=True); q.add_argument("--task", choices=["task1", "task2", "task3"], required=True)
    q = sub.add_parser("validate"); q.add_argument("--task", choices=["1", "2", "3"], required=True); q.add_argument("--input", required=True)
    return p


def validate(path, task):
    a = _anndata().read_h5ad(path)
    if a.n_obs == 0 or a.n_vars == 0: raise ValueError("empty prediction")
    x = np.asarray(a.X.toarray() if hasattr(a.X, "toarray") else a.X)
    if x.dtype.kind not in "fc" or not np.isfinite(x).all() or (x < 0).any(): raise ValueError(".X must be finite, numeric and non-negative")
    if len(set(map(str, a.var_names))) != a.n_vars: raise ValueError("var_names must be unique")
    if task in {"2", "3"}:
        if "spatial_3D" not in a.obsm: raise ValueError("spatial_3D missing")
        xyz = np.asarray(a.obsm["spatial_3D"])
        if xyz.shape != (a.n_obs, 3) or not np.isfinite(xyz).all(): raise ValueError("spatial_3D must be finite n×3")
    if "celltype" in a.obs: raise ValueError("submission must not include celltype")
    print(f"OK task={task} cells={a.n_obs} genes={a.n_vars}")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "compile-knowledge":
        kb = compile_bundle(args.output, args.sources, args.manifest); print(f"OK resources={len(kb.manifest['resources'])} output={args.output}"); return
    if args.command == "audit-knowledge":
        kb = KnowledgeBundle.load(args.knowledge, args.task); print(f"OK task={args.task} resources={len(kb.manifest['resources'])}"); return
    ad = _anndata()
    if args.command == "validate": return validate(args.input, args.task)
    params = vars(args).copy()
    if args.command in {"task1", "task2"}:
        if len(args.inputs) != len(args.times): raise SystemExit("--inputs and --times lengths differ")
        data = [ad.read_h5ad(p) for p in args.inputs]
        fn = predict_temporal if args.command == "task1" else predict_spatial
        result = fn(data, args.times, args.target_time, args.go, args.cells, args.metacell_size, args.seed,
                    knowledge_path=args.knowledge)
        if args.command == "task1": x, genes, modules = result; xyz = None
        else: x, xyz, genes, modules = result
        inputs = args.inputs + ([args.go] if args.go else [str(Path(args.knowledge) / "manifest.json")]); metadata = {"target_time": args.target_time}
    else:
        data = [ad.read_h5ad(p) for p in (args.wt_train, args.ko_train, args.wt_target)]
        x, xyz, genes, modules = predict_perturbation(*data, args.source_gene, args.target_gene, args.go,
                                                       args.cells, args.metacell_size, args.seed,
                                                       knowledge_path=args.knowledge, causal_strength=args.causal_strength,
                                                       communication_strength=args.communication_strength)
        inputs = [args.wt_train, args.ko_train, args.wt_target] + ([args.go] if args.go else [str(Path(args.knowledge) / "manifest.json")])
        metadata = {"perturbation": args.target_gene + "_KO"}
    _write(args.output, x, genes, xyz, metadata); validate(args.output, args.command[-1])
    write_run_record(args.output, args.command, inputs, params, {"cells": len(x), "genes": len(genes), "go_modules": len(modules)})


if __name__ == "__main__": main()
