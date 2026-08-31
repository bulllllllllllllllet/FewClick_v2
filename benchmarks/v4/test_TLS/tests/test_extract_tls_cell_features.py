from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from benchmarks.v4.test_TLS import extract_tls_cell_features


class OpenWsiTest(unittest.TestCase):
    @patch.object(extract_tls_cell_features, 'get_slide')
    def test_kfb_uses_shared_reader(self, get_slide: Mock) -> None:
        handle = object()
        get_slide.return_value = handle

        self.assertIs(extract_tls_cell_features.open_wsi(Path('case.KFB')), handle)
        get_slide.assert_called_once_with('case.KFB')

    @patch.object(extract_tls_cell_features, 'get_slide')
    def test_openslide_format_uses_shared_reader(self, get_slide: Mock) -> None:
        handle = object()
        get_slide.return_value = handle

        self.assertIs(extract_tls_cell_features.open_wsi(Path('case.svs')), handle)
        get_slide.assert_called_once_with('case.svs')


if __name__ == '__main__':
    unittest.main()
