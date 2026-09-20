import unittest

from benchmarks.v5.phase_5_prompt_encoder.src.extensible_task_contract import ExtensibleTaskManifest


def row(sample_id, split, group, footprint, *, case="same_case", source="slide_a"):
    return {
        "sample_id": sample_id, "source_id": source, "case_id": case, "section_id": "section_1",
        "spatial_group_id": group, "task_id": "structure.vessel", "split": split,
        "supervision_level": "pixel", "pairing_level": "same_region", "coordinate_system": "slide_a_um",
        "source_footprint": footprint, "context_footprint": footprint,
    }


class ExtensibleTaskContractTest(unittest.TestCase):
    def spec(self, rows):
        return {"schema_version": "extensible_prompt_task_v1", "ontology_version": "fixture-v1",
                "split_buffer_um": 5.0, "samples": rows}

    def test_same_case_cross_split_is_allowed_and_audited(self):
        manifest = ExtensibleTaskManifest(self.spec([
            row("train", "train", "g_train", [0, 0, 20, 20]),
            row("val", "val", "g_val", [40, 0, 60, 20]),
        ]))
        self.assertEqual(manifest.audit()["cases_crossing_splits"], ["same_case"])
        self.assertEqual(manifest.audit()["spatial_groups"], 2)

    def test_spatial_group_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, "spatial_group_id"):
            ExtensibleTaskManifest(self.spec([
                row("train", "train", "paired", [0, 0, 20, 20]),
                row("val", "val", "paired", [40, 0, 60, 20]),
            ]))

    def test_overlapping_context_is_rejected_even_with_different_groups(self):
        with self.assertRaisesRegex(ValueError, "context footprints overlap"):
            ExtensibleTaskManifest(self.spec([
                row("train", "train", "g_train", [0, 0, 20, 20]),
                row("val", "val", "g_val", [24, 0, 44, 20]),
            ]))

    def test_different_coordinate_sources_are_not_compared_as_if_registered(self):
        manifest = ExtensibleTaskManifest(self.spec([
            row("train", "train", "g_train", [0, 0, 20, 20], source="slide_a"),
            row("val", "val", "g_val", [0, 0, 20, 20], source="slide_b"),
        ]))
        self.assertEqual(manifest.audit()["samples"], 2)


if __name__ == "__main__":
    unittest.main()
