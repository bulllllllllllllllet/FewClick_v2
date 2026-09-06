"""Phikon-v2 feature extraction and H&E-only cell adapter for FewClick v5.

This module is intentionally independent of the frozen v4 CTransPath/XCellFormer
implementation.  It accepts RGB PIL crops and exposes a 1024-d CLS feature
contract, followed by a trainable adapter for downstream cell supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn


PHIKON_V2_FEATURE_DIM = 1024
MIN_TRANSFORMERS_VERSION = (4, 31)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(".")[:2])


def _load_transformers():
    import transformers

    if _version_tuple(transformers.__version__) < MIN_TRANSFORMERS_VERSION:
        required = ".".join(map(str, MIN_TRANSFORMERS_VERSION))
        raise RuntimeError(
            f"Phikon-v2 needs transformers >= {required}; found {transformers.__version__}. "
            "Use the v5-only dependency overlay documented in requirements-phikon-v2.txt; "
            "do not upgrade the frozen v4 environment in place."
        )
    return transformers.AutoImageProcessor, transformers.AutoModel


@dataclass(frozen=True)
class PhikonV2Contract:
    model_dir: Path
    feature_dim: int = PHIKON_V2_FEATURE_DIM
    input_size: int = 224
    magnification: str = "20x"
    modality: str = "H&E RGB only"


class PhikonV2Encoder(nn.Module):
    """Frozen-or-trainable Phikon-v2 encoder returning CLS-1024 features."""

    def __init__(self, model_dir: str | Path, *, device: str | torch.device = "cpu", dtype: torch.dtype | None = None):
        super().__init__()
        self.contract = PhikonV2Contract(Path(model_dir))
        if not self.contract.model_dir.is_dir():
            raise FileNotFoundError(f"Phikon-v2 model directory does not exist: {self.contract.model_dir}")
        AutoImageProcessor, AutoModel = _load_transformers()
        self.processor = AutoImageProcessor.from_pretrained(str(self.contract.model_dir), local_files_only=True)
        load_kwargs = {"local_files_only": True}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        self.backbone = AutoModel.from_pretrained(str(self.contract.model_dir), **load_kwargs)
        self.backbone.to(device)

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        values = outputs.last_hidden_state[:, 0, :]
        if values.ndim != 2 or values.shape[1] != PHIKON_V2_FEATURE_DIM:
            raise RuntimeError(f"unexpected Phikon-v2 CLS shape: {tuple(values.shape)}")
        if not torch.isfinite(values).all():
            raise RuntimeError("Phikon-v2 produced non-finite CLS features")
        return values

    def preprocess(self, crops: Iterable[Image.Image], device: str | torch.device) -> torch.Tensor:
        images = [crop.convert("RGB") for crop in crops]
        if not images:
            raise ValueError("at least one H&E RGB crop is required")
        batch = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return batch.to(device)

    @torch.inference_mode()
    def encode_pil(self, crops: Iterable[Image.Image], device: str | torch.device) -> torch.Tensor:
        return self.forward(self.preprocess(crops, device))


class CellFeatureAdapter(nn.Module):
    """Trainable 1024-d Phikon CLS to compact v5 H&E cell embedding adapter."""

    def __init__(self, output_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        if output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("output_dim and hidden_dim must be positive")
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.LayerNorm(PHIKON_V2_FEATURE_DIM),
            nn.Linear(PHIKON_V2_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, phikon_cls: torch.Tensor) -> torch.Tensor:
        if phikon_cls.ndim != 2 or phikon_cls.shape[1] != PHIKON_V2_FEATURE_DIM:
            raise ValueError(f"expected [batch,{PHIKON_V2_FEATURE_DIM}], got {tuple(phikon_cls.shape)}")
        result = self.network(phikon_cls)
        if not torch.isfinite(result).all():
            raise RuntimeError("cell adapter produced non-finite features")
        return result
