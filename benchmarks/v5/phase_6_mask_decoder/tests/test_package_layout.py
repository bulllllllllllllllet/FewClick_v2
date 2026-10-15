"""Protect existing imports, checkpoint class paths, and source-hash lookup."""
import importlib
import unittest
from pathlib import Path
from benchmarks.v5.phase_6_mask_decoder import _MODULE_GROUPS, module_source


class PackageLayoutTests(unittest.TestCase):
    def test_all_public_modules_resolve_to_single_registered_source(self):
        names = [name for modules in _MODULE_GROUPS.values() for name in modules]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            with self.subTest(module=name):
                module = importlib.import_module('benchmarks.v5.phase_6_mask_decoder.' + name)
                self.assertEqual(Path(module.__file__).resolve(), module_source(name).resolve())
                self.assertTrue(module_source(name).is_file())

    def test_checkpoint_model_class_paths_remain_public_module_names(self):
        for module_name, class_name in [('dense_prompt_decoder', 'DensePromptDecoder'),
                                        ('continue_dense_boundary', 'DenseContextModel'),
                                        ('point_boundary_detail', 'PointBoundaryDetail'),
                                        ('iterative_decoder', 'IterativePromptDecoder')]:
            module = importlib.import_module('benchmarks.v5.phase_6_mask_decoder.' + module_name)
            self.assertEqual(getattr(module, class_name).__module__, module.__name__)

    def test_unknown_source_name_is_rejected(self):
        with self.assertRaises(KeyError):
            module_source('not_a_registered_module')


if __name__ == '__main__':
    unittest.main()
