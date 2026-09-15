import unittest

import numpy as np
import pandas as pd

from benchmarks.v5.tools.build_crc_manual_review_package import (
    CLASSES, add_stratum, html_page, panel_image, stratified_sample,
)


class ManualReviewPackageTest(unittest.TestCase):
    def test_risk_strata_are_mutually_exclusive(self):
        frame = pd.DataFrame({"geometry_review_pass": [True, False, True],
                              "stable_single_marker": [True, True, False]})
        self.assertEqual(add_stratum(frame).review_stratum.tolist(),
                         ["joint_diagnostic_pass", "geometry_risk", "label_rule_risk"])

    def test_sampling_is_unique_balanced_and_treg_census(self):
        rows = []
        for case in ["a", "b", "c"]:
            for label in CLASSES:
                for i in range(30 if label != "Treg" else 3):
                    rows.append({"case": case, "class": label, "patch": "x0_y0", "he_id": i,
                                 "geometry_review_pass": i % 3 != 1,
                                 "stable_single_marker": i % 3 == 0})
        sample = stratified_sample(pd.DataFrame(rows), per_case_class=6, seed=7)
        self.assertEqual(len(sample), 4 * 3 * 6 + 3 * 3)
        self.assertEqual(sample.review_id.nunique(), len(sample))
        self.assertEqual((sample["class"] == "Treg").sum(), 9)
        self.assertTrue((sample.selection_probability_within_candidate_pool > 0).all())

    def test_panel_and_html_are_self_contained(self):
        he = np.zeros((65, 65, 3), np.uint8)
        stack = np.ones((7, 65, 65), np.uint16) * 100
        mask = np.zeros((65, 65), np.int32); mask[28:37, 28:37] = 1
        row = type("Row", (), {"x": 32., "y": 32., "he_id": 1, "dapi_id": 1})()
        image = panel_image(he, stack, {name: i for i, name in enumerate(["DAPI", "CD4", "CD8", "CD20", "CD163", "FOXP3", "SULF1"])}, mask, mask, row)
        self.assertEqual(image.size, (1040, 154))
        html = html_page([{"review_id": "R0001", "image": "images/R0001.jpg", "auto_label": "Treg",
                           "case": "case", "stratum": "risk", "intensities": "CD4=1"}])
        self.assertIn("localStorage", html)
        self.assertIn("IDENTITY_UNCERTAIN", html)


if __name__ == "__main__":
    unittest.main()
