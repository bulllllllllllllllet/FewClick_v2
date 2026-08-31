from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmarks.v4.whole_slide_inference.infer_wsi import SlideReader


class _FakeSlide:
    dimensions = (64, 48)

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], int, tuple[int, int]]] = []

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        self.calls.append((location, level, size))
        return Image.new("RGBA", size, (11, 22, 33, 255))


class SlideReaderTest(unittest.TestCase):
    def test_every_supported_backend_uses_vipa_reader_contract(self) -> None:
        for suffix in (".kfb", ".tiff"):
            with self.subTest(suffix=suffix):
                slide = _FakeSlide()
                with patch(
                    "benchmarks.v4.whole_slide_inference.infer_wsi.get_slide",
                    return_value=slide,
                ) as get_slide:
                    reader = SlideReader(Path(f"/wsis/sample{suffix}"))
                    image = reader.read(
                        {
                            "x_level0": 7,
                            "y_level0": 9,
                            "width_level0": 16,
                            "height_level0": 16,
                            "width_10x": 8,
                            "height_10x": 8,
                        }
                    )

                get_slide.assert_called_once_with(f"/wsis/sample{suffix}")
                self.assertEqual(slide.calls, [((7, 9), 0, (16, 16))])
                self.assertEqual(image.shape, (8, 8, 3))
                np.testing.assert_array_equal(image[0, 0], np.array([11, 22, 33], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
