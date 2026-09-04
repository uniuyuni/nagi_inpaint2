"""Stage A — 低周波補完 (SPEC §4.1)。

**役割**: 穴を低周波で埋めて、Stage B の照合にクエリを与える。

穴の内部画素はパッチが丸ごと穴に覆われるため照合コストが定義できず、
**照合可能なブロックは 36% しかない**。残りは縁からの伝播で決まり「選ばれて」いない。
低周波で埋めると照合可能が 100% になり、実測で **+7.4 dB**（21.65 → 29.06 dB）。
Stage A は「あれば良い」ではなく、**無ければ Stage B が動かない**。

**必要なのは低周波だけ、という事実が設計を大きく単純にする**

Stage B に渡すクエリは 1/8 に縮小した低周波である。つまり Stage A は
**1/8 解像度で動けば足りる** (512² タイルなら 64²、長辺2048の画像なら 256²)。

  - 高周波を要求しないので GAN も知覚損失も不要。L1 + 勾配損失で足りる
  - 低解像度なので軽い。1/8 で動かせば計算量は 1/64
  - 天井 (神託 33.27 dB) まで余裕があるので、品質要求も厳しくない

**規模**: base=40 で約 5M params (§4.1 の想定)。base=64 では 14.4M になり、
低周波補完には過大だった。天井 (神託 33.27 dB) まで余裕があるので品質要求は厳しくない。

**構造**: FFC (Fourier Convolution) ベースの U-Net。
P1 で MPS の複素数FFTが CPU比 5.13倍・autograd 可と確認済み (§10.1)。
低解像度で大域受容野を得るのに FFT は効率が良い。

**正規化は InstanceNorm。BatchNorm を使ってはならない。**
入力の穴はゼロで埋まっているため、バッチ統計がマスク面積に依存して揺れる。
実測でも BatchNorm の running_var が初期値から **340倍** に膨れ上がっていた
(重み自体の変化は max 0.51 だったので、この値は学習量ではなく統計の暴走)。
インペインティングでは InstanceNorm か正規化なしが定石である。
  https://discuss.pytorch.org/t/masked-batchnorm-for-variable-length-input/107286
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SHRINK = 8            # Stage B が使う低周波の縮小率


class SpectralBlock(nn.Module):
    """周波数領域での 1x1 畳み込み。1層で大域受容野が得られる。"""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch * 2, ch * 2, 1)
        self.bn = nn.InstanceNorm2d(ch * 2, affine=True)

    def forward(self, x):
        b, c, h, w = x.shape
        f = torch.fft.rfft2(x, norm="ortho")
        f = torch.cat([f.real, f.imag], 1)
        f = F.relu(self.bn(self.conv(f)))
        re, im = f.chunk(2, 1)
        return torch.fft.irfft2(torch.complex(re, im), s=(h, w), norm="ortho")


class FFC(nn.Module):
    """局所枝 (通常の畳み込み) と大域枝 (周波数領域) を併走させる。"""

    def __init__(self, cin: int, cout: int, ratio: float = 0.5):
        super().__init__()
        self.cg = int(cout * ratio)
        self.cl = cout - self.cg
        self.local = nn.Conv2d(cin, self.cl, 3, padding=1)
        self.to_g = nn.Conv2d(cin, self.cg, 3, padding=1)
        self.spec = SpectralBlock(self.cg)
        self.norm = nn.InstanceNorm2d(cout, affine=True)

    def forward(self, x):
        l = self.local(x)
        g = self.spec(self.to_g(x))
        return F.relu(self.norm(torch.cat([l, g], 1)))


class StageA(nn.Module):
    """低周波補完。入力・出力とも 1/SHRINK 解像度。

    入力: 知覚空間3ch (穴は 0) + マスク1ch
    出力: 知覚空間3ch (穴の中身)
    """

    def __init__(self, base: int = 40, depth: int = 3, n_ffc: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(4, base, 3, padding=1), nn.ReLU())
        enc, ch = [], base
        for _ in range(depth):
            enc.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                                     nn.InstanceNorm2d(ch * 2, affine=True), nn.ReLU()))
            ch *= 2
        self.enc = nn.ModuleList(enc)
        self.mid = nn.Sequential(*[FFC(ch, ch) for _ in range(n_ffc)])
        dec, chans = [], []
        for _ in range(depth):
            chans.append(ch)
            dec.append(nn.Sequential(nn.Conv2d(ch + ch // 2, ch // 2, 3, padding=1),
                                     nn.InstanceNorm2d(ch // 2, affine=True), nn.ReLU()))
            ch //= 2
        self.dec = nn.ModuleList(dec)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, img, mask):
        x = torch.cat([img * (1 - mask), mask], 1)
        x = self.stem(x)
        skips = [x]
        for e in self.enc:
            x = e(x)
            skips.append(x)
        x = self.mid(x)
        for d in self.dec:
            skip = skips[-2]
            skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = d(torch.cat([x, skip], 1))
        return self.out(x)


def low_target(img: torch.Tensor, shrink: int = SHRINK) -> torch.Tensor:
    """学習ターゲット: 低域通過した正解。高周波は要求しない。"""
    return F.interpolate(img, scale_factor=1.0 / shrink, mode="area")


@torch.no_grad()
def make_query(model: StageA, img: torch.Tensor, mask: torch.Tensor,
               shrink: int = SHRINK) -> torch.Tensor:
    """Stage B に渡すクエリを作る。穴を低周波補完で埋めた全解像度画像。"""
    lo_img = F.interpolate(img, scale_factor=1.0 / shrink, mode="area")
    lo_mask = (F.interpolate(mask, scale_factor=1.0 / shrink, mode="area") > 0.05).float()
    pred = model(lo_img, lo_mask)
    filled = torch.where(lo_mask > 0.5, pred, lo_img)
    up = F.interpolate(filled, size=img.shape[-2:], mode="bilinear",
                       align_corners=False)
    return torch.where(mask > 0.5, up, img)
