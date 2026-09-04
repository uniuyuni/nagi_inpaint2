"""Real-ESRGAN による拡大 (L4 の復元手段)。

**なぜ必要か**

512² に収まらない対象 (実測 31%) は縮小してモデルに渡すしかない。
縮小 2 倍で**高周波 (細かい起伏) が 26% しか残らない** (実測)。
そのままでは穴の中だけツルツルになる (grain 0.781 -> 0.622)。

初版は手作りの「高周波移植」を書いたが、これは **R17 違反**だった
(自作の前に既存を試していない)。学習済みの超解像モデルなら、
統計を合わせるだけでなく**もっともらしい細部を作れる**。

**モデル**: Real-ESRGAN x2plus (`RealESRGAN_x2plus.pth`, 67MB, 16.7M params)。
公式リポジトリ `xinntao/Real-ESRGAN` は **BSD-3-Clause**。

**実装について**: `basicsr` パッケージは依存が重く、
新しい torchvision との既知の非互換 (`functional_tensor`) があるので入れない。
RRDBNet は標準的な構成なので直接書き、
**`load_state_dict(strict=True)` が通ることで正しさを検証する**
(702 テンソルのキーと形が全一致しなければ落ちる)。

**入力域**: Real-ESRGAN は 8bit sRGB で学習されている。
知覚空間の値をそのまま渡さず、呼び出し側で [0,1] に正規化すること
(YOEO で「暗く低コントラストな入力で拡散モデルが崩れる」を実測している)。
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = "models/realesrgan/RealESRGAN_x2plus.pth"


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf: int, gc: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN の生成器。`scale=2` は先に pixel_unshuffle するので入力 12ch。"""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 2,
                 nf: int = 64, num_block: int = 23, gc: int = 32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(
            F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(
            F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class RealESRGAN:
    """`up(x_hwc_01) -> x_hwc_01` (2 倍)。入力は [0,1] の sRGB 相当。"""

    def __init__(self, path: str = WEIGHTS, device: str = "mps",
                 dtype=torch.float32):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} が無い。以下で取得:\n"
                "  curl -fL -o models/realesrgan/RealESRGAN_x2plus.pth \\\n"
                "    https://github.com/xinntao/Real-ESRGAN/releases/"
                "download/v0.2.1/RealESRGAN_x2plus.pth")
        w = torch.load(path, map_location="cpu", weights_only=False)
        sd = w.get("params_ema") or w.get("params") or w
        self.net = RRDBNet(scale=2)
        # **strict=True が通ることが実装の検証になる** (702 テンソル全一致)
        self.net.load_state_dict(sd, strict=True)
        self.net = self.net.to(device, dtype).eval()
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))[None])
        t = t.to(self.device, self.dtype).clamp(0, 1)
        y = self.net(t).clamp(0, 1)
        return y[0].permute(1, 2, 0).float().cpu().numpy()
