"""Adversarial manifest checks and loader-to-optimizer integration fixtures."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmarks.v5.phase_3_cell_region.src.semantic_contract import SemanticDataset, sha256_file
from benchmarks.v5.phase_3_cell_region.src.semantic_cell_region import SemanticCellToRegion
from benchmarks.v5.phase_3_cell_region.src.semantic_training import evaluate, training_loss


class ContractTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        cells = rng.normal(size=(3, 74)).astype(np.float32)
        cells[:, :2] = 0.5
        cells[:, 3:10] = 0
        cells[:, 3] = 1
        self.arrays = dict(region_tokens=rng.normal(size=(2, 256)).astype(np.float32), cells=cells,
                           cell_valid=np.ones(3, bool), assignment_at_cells=np.full((2, 3), 0.5, np.float32),
                           region_area_um2=np.full(2, 1000., np.float32), cell_ids=np.arange(3),
                           identity_allowed=np.eye(3, dtype=bool), identity_valid=np.ones(3, bool),
                           tissue_target=np.eye(2, dtype=np.float32), tissue_valid=np.ones(2, bool),
                           composition_target=np.full((2, 3), 1 / 3, np.float32), composition_valid=np.ones(2, bool))
        self.spec = {"schema_version": "semantic_phase3_v1", "identity_classes": ["a", "b", "c"],
                     "tissue_classes": ["t0", "t1"], "provenance": {
                         "feature_kind": "he_only_xcell_reg64", "sampling": "complete_detected_inventory",
                         "nuclei_class_source": "unavailable", "registration_qc": "same_he_annotation",
                         **{k: "fixture" for k in ("detector_version", "xcell_checkpoint_sha256", "phase2_checkpoint_sha256",
                                                    "ontology_version", "coordinate_version", "label_source")}},
                     "samples": []}
        for split in ("train", "val"):
            path = self.root / f"{split}.npz"
            np.savez(path, **self.arrays)
            self.spec["samples"].append(dict(sample_id=split, case_id=split, organ="fixture", split=split,
                                               path=path.name, sha256=sha256_file(path)))
        self.manifest = self.root / "manifest.json"
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.spec))

    def replace_train_arrays(self):
        path = self.root / "train.npz"
        np.savez(path, **self.arrays)
        self.spec["samples"][0]["sha256"] = sha256_file(path)
        self.write_manifest()

    def test_loader_optimizer_evaluation_both_stages(self):
        train = SemanticDataset(self.manifest, "train")
        val = SemanticDataset(self.manifest, "val")
        model = SemanticCellToRegion(3, 2)
        for stage in ("identity", "fusion"):
            model.set_stage(stage)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            item = train[0]
            output = model(**item["inputs"])
            loss, count = training_loss(model, output, item["targets"], item["inputs"]["cell_valid"])
            self.assertGreater(count, 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metrics = evaluate(model, val, torch.device("cpu"))
            self.assertIsNotNone(metrics["patient_macro_f1"])
            self.assertIsNotNone(metrics["cases"]["val"]["brier"])

    def test_case_leakage_rejected(self):
        self.spec["samples"][1]["case_id"] = "train"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            SemanticDataset(self.manifest, "train")

    def test_explicit_pooled_protocol_allows_internal_case_overlap(self):
        self.spec["samples"][1]["case_id"] = "train"
        self.spec["provenance"]["split_protocol"] = "pooled_case_overlapping_patch_internal_v1"
        self.write_manifest()
        self.assertEqual(len(SemanticDataset(self.manifest, "train")), 1)

    def test_truth_class_provenance_rejected(self):
        self.spec["provenance"]["nuclei_class_source"] = "ground_truth"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "GT nuclei"):
            SemanticDataset(self.manifest, "train")

    def test_hash_tamper_rejected(self):
        dataset = SemanticDataset(self.manifest, "train")
        self.arrays["cells"][0, 10] += 1
        self.replace_train_arrays()
        with self.assertRaisesRegex(ValueError, "checksum"):
            dataset[0]

    def test_region_qc_cannot_authorize_cell_labels(self):
        self.spec["provenance"]["registration_qc"] = "region_correspondence_verified"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            SemanticDataset(self.manifest, "train")[0]

    def test_negative_distribution_rejected(self):
        self.arrays["tissue_target"][0] = [-1, 2]
        self.replace_train_arrays()
        with self.assertRaisesRegex(ValueError, "target distribution"):
            SemanticDataset(self.manifest, "train")[0]

    def test_missing_identity_skips_optimization(self):
        self.arrays["identity_valid"][:] = False
        self.replace_train_arrays()
        model = SemanticCellToRegion(3, 2)
        model.set_stage("identity")
        item = SemanticDataset(self.manifest, "train")[0]
        loss, count = training_loss(model, model(**item["inputs"]), item["targets"], item["inputs"]["cell_valid"])
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.detach()), 0.)


if __name__ == "__main__":
    unittest.main()
