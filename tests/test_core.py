import numpy as np

from ve.core import GeneModules, ModuleDecoder, build_metacells, forecast, sample_state
from ve.knowledge import KnowledgeBundle


def test_go_roundtrip_and_no_gene_drop(tmp_path):
    path = tmp_path / "go.tsv"
    path.write_text("gene\tmodule\nA\tdev\nB\tdev\nC\tsignal\n")
    x = np.random.default_rng(0).random((100, 5), dtype=np.float32)
    gm = GeneModules.from_tsv(["A", "B", "C", "D", "E"], path, unannotated_chunk=2)
    assert gm.weights.shape[0] == 5
    z = gm.encode(x); dec = ModuleDecoder.fit(z, x)
    assert dec.decode(z[:10], np.random.default_rng(1), noise=0).shape == (10, 5)


def test_metacell_forecast_samples_distribution():
    rng = np.random.default_rng(2)
    x1 = rng.random((80, 4)); x2 = np.maximum(x1 + .2, 0)
    labels = np.repeat(["a", "b"], 40)
    s1 = build_metacells(x1, labels, None, 10, 2)
    s2 = build_metacells(x2, labels, None, 10, 3)
    pred = forecast([s1, s2], [1, 2], 3)
    sampled, xyz = sample_state(pred, 71, rng)
    assert sampled.shape == (71, 4) and xyz is None
    assert np.isfinite(sampled).all() and (sampled >= 0).all()


def test_interpolation_uses_bracketing_stages():
    labels = ["a"] * 20
    stages = [build_metacells(np.full((20, 2), v), labels, None, 100, i) for i, v in enumerate([0., 2., 20.])]
    pred = forecast(stages, [1., 3., 10.], 2.)
    assert np.allclose(pred.expression, 1.)


def test_signed_ko_and_leakage_audit(tmp_path):
    bundle = tmp_path / "kb"; bundle.mkdir()
    (bundle / "manifest.json").write_text('{"resources":[{"name":"reg","version":"1","url":"x","license":"CC","species":"Mus musculus","kind":"signed_regulation","measured_expression":false}]}')
    (bundle / "regulatory.tsv").write_text("regulator\ttarget\tsign\tconfidence\tsource\nA\tB\t1\t1\ttest\nA\tC\t-1\t1\ttest\n")
    kb = KnowledgeBundle.load(bundle, "task3")
    effect = kb.causal_gene_effect(["A", "B", "C"], "A")
    assert effect[1] < 0 and effect[2] > 0
    (bundle / "manifest.json").write_text('{"resources":[{"name":"bad","version":"1","url":"x","license":"x","species":"mouse","kind":"expression","measured_expression":true,"developmental_stages":["E12.5"],"genotypes":[]}]}')
    try:
        KnowledgeBundle.load(bundle, "task1")
        assert False, "held-out expression should be rejected"
    except ValueError as e:
        assert "held-out leakage" in str(e)


def test_evidence_family_deduplication(tmp_path):
    bundle = tmp_path / "kb2"; bundle.mkdir()
    (bundle / "manifest.json").write_text('{"resources":[{"name":"net","version":"1","url":"x","license":"CC","species":"mouse","kind":"network","measured_expression":false}]}')
    (bundle / "associations.tsv").write_text(
        "gene_a\tgene_b\trelation\tgroup\tconfidence\tevidence_family\tspecies_match\tcontext_match\tdirectness\tsource\n"
        "A\tB\tphysical\tx\t0.8\tphysical\t1\t1\t1\tBioGRID\n"
        "A\tB\tphysical\tx\t0.7\tphysical\t1\t1\t1\tSTRING-imported-BioGRID\n"
        "A\tB\tpathway\tx\t0.5\tcurated_pathway\t1\t1\t1\tReactome\n")
    kb = KnowledgeBundle.load(bundle)
    # max(0.8, 0.7) inside physical family, then noisy-OR with independent pathway evidence.
    assert np.isclose(kb.association_consensus(["A", "B"])[0, 1], 0.9)
