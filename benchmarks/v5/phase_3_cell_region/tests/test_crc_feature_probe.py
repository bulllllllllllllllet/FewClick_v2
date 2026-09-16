"""Pure invariants of the CRC candidate feature preparation path."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmarks.v5.tools.prepare_crc_he_feature_probe import broad_class, centers, relocate
from benchmarks.v5.phase_3_cell_region.src.crc_sampling import FINE_CLASSES, fine_class, choose_balanced_patches, balanced_table, quota_patch_table


class CRCFeatureProbeTest(unittest.TestCase):
    def test_explicit_relocation(self):
        root = Path("/nfs-medical5/hwx/panel2_cell_level_registration")
        old = "/home/hwx/panel2_cell_level_registration/2022-39941E/transforms/x0_y0.json"
        self.assertEqual(relocate(old, root), root / "2022-39941E/transforms/x0_y0.json")
        with self.assertRaises(ValueError):
            relocate("/home/another/data.json", root)

    def test_identity_and_state_separated(self):
        self.assertEqual(broad_class("CD4_T_SULF1"), 0)
        self.assertEqual(broad_class("Treg"), 0)
        self.assertEqual(broad_class("B_cell_SULF1"), 1)
        self.assertEqual(broad_class("CD163_Macrophage"), 2)
        for value in ("Undefined", "SULF1_only", "FOXP3_only_Unresolved_SULF1"):
            self.assertEqual(broad_class(value), -1)

    def test_centers_are_y_x_sorted_but_export_x_y(self):
        mask = np.zeros((8, 8), np.int32)
        mask[1:3, 4:6] = 9
        mask[5:7, 0:2] = 3
        ids, xy = centers(mask)
        np.testing.assert_array_equal(ids, [9, 3])
        np.testing.assert_allclose(xy, [[4.5, 1.5], [.5, 5.5]])

    def test_empty_mask(self):
        ids, xy = centers(np.zeros((4, 4), np.int32))
        self.assertEqual(len(ids), 0)
        self.assertEqual(xy.shape, (0, 2))

    def test_fine_labels_keep_treg_separate(self):
        for index, label in enumerate(FINE_CLASSES):
            self.assertEqual(fine_class(label + "_SULF1"), index)
        self.assertEqual(fine_class("Undefined"), -1)
        self.assertNotEqual(fine_class("Treg"), fine_class("CD4_T"))

    def test_balanced_selection_includes_rare_class_without_duplicates(self):
        counts = pd.DataFrame([[0, 0, 1, 0, 0], [20, 10, 0, 5, 50], [0, 0, 0, 0, 100]],
                              index=["x0_y0", "x4096_y0", "x8192_y0"], columns=FINE_CLASSES)
        selected = choose_balanced_patches(counts, 2)
        self.assertEqual(selected[0], "x0_y0")
        self.assertEqual(len(set(selected)), 2)

    def test_balanced_table_preserves_patch_across_chunks_and_unknown_cells(self):
        frame = pd.DataFrame({"patch_name": ["x0_y0"] * 31,
                              "class": ["Treg"] + ["Undefined"] * 30,
                              "cell_id": np.arange(31), "x": 0, "y": 0,
                              "registered_he_patch": "image.png", "local_transform": "transform.json"})
        chunks = [frame.iloc[:15].copy(), frame.iloc[15:].copy()]
        with patch("benchmarks.v5.phase_3_cell_region.src.crc_sampling.pd.read_csv",
                   side_effect=[iter(chunks), iter(chunks)]):
            groups, report = balanced_table("unused.tsv", 1, 40)
        self.assertEqual(len(groups[0][1]), 31)
        self.assertEqual(report["source_rows"], 31)
        self.assertEqual(report["source_class_counts"]["Treg"], 1)
        self.assertEqual(report["eligible_patches"], 1)

    def test_quota_selector_reaches_available_targets(self):
        labels = (["CD4_T"] * 10 + ["Treg"] * 2 + ["Undefined"] * 19,
                  ["CD8_T"] * 10 + ["B_cell"] * 10 + ["CD163_Macrophage"] * 10 + ["Undefined"])
        frame = pd.concat([pd.DataFrame({"patch_name": f"x{index * 2048}_y0", "class": values}) for index, values in enumerate(labels)], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cells.tsv"; frame.to_csv(path, sep="\t", index=False)
            report = quota_patch_table(path, {"CD4_T": 10, "CD8_T": 10, "Treg": 2, "B_cell": 10, "CD163_Macrophage": 10}, 40)
        self.assertEqual(len(report["selected_patches"]), 2)
        self.assertEqual(report["achieved_source_candidates"]["Treg"], 2)


if __name__ == "__main__":
    unittest.main()
