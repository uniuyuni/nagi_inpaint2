"""LaMa (big-lama) の推論。**製品側の経路に評価用コードを持ち込まない。**

元は `eval/lama_baseline.py` にあったが、あのファイルは `eval.metrics` /
`eval.validate_stage_d` / `data.masks` を import する評価用スクリプトである。
ShadeWave から呼ぶ経路がそれらに依存するのは筋が悪いので、
**クラスだけをここへ移した** (2026-08-30)。
`eval/lama_baseline.py` はここから import するので従来の使い方は変わらない。

重み: `models/big-lama.pt` (Apache-2.0)。パスは `nagi_inpaint2.paths.resolve`
で解決するので、**呼び出し元の作業ディレクトリに依存しない**。
"""
from __future__ import annotations

import os

import numpy as np
import torch

from nagi_inpaint2.paths import resolve

MODEL = "models/big-lama.pt"
NORM_PCT = 99.9      # タイルごとの正規化基準。クリップを避ける


class Lama:
    def __init__(self, path: str = MODEL, device: str = "mps"):
        path = resolve(path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} がない。以下で取得すること:\n"
                "  curl -fL -o models/big-lama.pt "
                "https://github.com/enesmsahin/simple-lama-inpainting/"
                "releases/download/v0.1.0/big-lama.pt")
        self.device = device
        self.m = torch.jit.load(path, map_location=device).eval()

    def __call__(self, p: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """知覚空間タイル (H,W,3) と mask (H,W) を受け、知覚空間で返す。"""
        k = max(1.0, float(np.percentile(p, NORM_PCT)))
        s = np.clip(p / k, 0.0, 1.0)
        x = torch.from_numpy(s.transpose(2, 0, 1)[None].astype(np.float32))
        m = torch.from_numpy(mask[None, None].astype(np.float32))
        with torch.inference_mode():
            y = self.m(x.to(self.device), m.to(self.device))
        return (y[0].permute(1, 2, 0).float().cpu().numpy() * k).astype(np.float32)
