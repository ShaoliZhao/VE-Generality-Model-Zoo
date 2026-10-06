import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ve.core import GeneModules, ModuleDecoder, build_metacells, forecast, sample_state
from ve.knowledge import KnowledgeBundle


class UnifiedModelTests(unittest.TestCase):
    def test_module_roundtrip_retains_every_gene(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "modules.tsv"
            path.write_text("gene\tmodule\nA\tdev\nB\tdev\nC\tsignal\n")
            x = np.random.default_rng(0).random((100, 5), dtype=np.float32)
            modules = GeneModules.from_tsv(["A", "B", "C", "D", "E"], path, unannotated_chunk=2)
            z = modules.encode(x)
            decoded = ModuleDecoder.fit(z, x).decode(z[:10], np.random.default_rng(1), noise=0)
            self.assertEqual(modules.weights.shape[0], 5)
            self.assertEqual(decoded.shape, (10, 5))

    def test_forecast_samples_a_valid_distribution(self):
        rng = np.random.default_rng(2)
        x1 = rng.random((80, 4)); x2 = np.maximum(x1 + 0.2, 0)
        labels = np.repeat(["a", "b"], 40)
        state = forecast([build_metacells(x1, labels, None, 10, 2), build_metacells(x2, labels, None, 10, 3)], [1, 2], 3)
        sampled, xyz = sample_state(state, 71, rng)
        self.assertEqual(sampled.shape, (71, 4)); self.assertIsNone(xyz)
        self.assertTrue(np.isfinite(sampled).all() and (sampled >= 0).all())

    def test_signed_ko_and_leakage_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {"resources": [{"name": "reg", "version": "1", "url": "x", "license": "CC", "species": "Mus musculus", "kind": "signed_regulation", "measured_expression": False}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "regulatory.tsv").write_text("regulator\ttarget\tsign\tconfidence\tsource\nA\tB\t1\t1\ttest\nA\tC\t-1\t1\ttest\n")
            effect = KnowledgeBundle.load(root, "task3").causal_gene_effect(["A", "B", "C"], "A")
            self.assertLess(effect[1], 0); self.assertGreater(effect[2], 0)
            manifest["resources"][0].update(measured_expression=True, developmental_stages=["E12.5"], genotypes=[])
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "held-out leakage"):
                KnowledgeBundle.load(root, "task1")


if __name__ == "__main__": unittest.main()
