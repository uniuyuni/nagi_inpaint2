"""Stage F — 全解像度 FFC 補完 (実現可能性検証用)。

**位置づけ**: Stage A (1/8 解像度・低周波専用) は測定の結果 LaMa に勝てないことが
確定した。低周波だけを渡してブロック照合 (Stage B) で高周波を復元する設計に
限界があるのか、それとも全解像度で直接学習すれば LaMa に迫れるのかを、
短時間の学習で見極めるための検証モデルがこの StageF である。
製品実装ではないので、正しく・速く測れることを優先する。

**StageA との違いは2点だけ**

  1. 入力を 1/8 に縮小しない (SHRINK 相当の処理を通さない)。全解像度のまま扱う。
  2. 全解像度のまま FFC (特に大域枝の FFT) を回すと計算量が 64 倍に膨らむ。
     LaMa と同じ考え方で、エンコーダで 1/4 まで stride 2 畳み込みを2段掛けて
     落とし、そこで FFC ブロックを通してからデコーダで全解像度に戻す。
     つまり FFT が回るのは 1/4 解像度の特徴マップに対してだけで、
     StageA (1/8) より荷は重いが全解像度直処理よりは軽い。

**正規化は StageA と同じく InstanceNorm。BatchNorm は使わない。**
穴がゼロ埋めされた入力ではバッチ統計がマスク面積に依存して揺れ、
BatchNorm の running_var が暴走する失敗が既に一度あった (StageA 開発時)。
同じ罠を Stage F でも踏まないよう、ブロック実装は StageA からそのまま流用する。

**forward の出力はマスク外もネットワークの生出力。** 呼び出し側 (損失関数や
推論コード) が `torch.where(mask>0.5, pred, img)` のような合成を行う前提で、
このモジュール自身はマスク処理をしない。理由は2つ:
  - 損失側で穴/既知領域を別重みで扱いたい (`train/train_stage_f.py` 参照)
  - 学習時に「既知領域も復元できる」ことを要求すると再構成の錨になり汎化が安定する
    (StageA の step_loss と同じ考え方。LaMa の設定に倣っている)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nagi_inpaint2.stage_a import FFC


class StageF(nn.Module):
    """全解像度 FFC 補完。入力・出力とも同じ解像度。

    入力: 知覚空間3ch (穴は 0) + マスク1ch
    出力: 知覚空間3ch (全解像度、マスク内外を問わずネットワークの生出力)

    構成: stem (全解像度) -> エンコーダ2段 (stride 2 x2, 全解像度の 1/4) ->
          FFC ブロック (1/4 解像度) -> デコーダ2段 (全解像度に復元)。
    """

    ENC_DEPTH = 2   # 1/4 まで落とす (2^2 = 4)

    def __init__(self, base: int = 32, n_ffc: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(4, base, 3, padding=1), nn.ReLU())
        enc, ch = [], base
        for _ in range(self.ENC_DEPTH):
            enc.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                                     nn.InstanceNorm2d(ch * 2, affine=True), nn.ReLU()))
            ch *= 2
        self.enc = nn.ModuleList(enc)
        self.mid = nn.Sequential(*[FFC(ch, ch) for _ in range(n_ffc)])
        dec = []
        for _ in range(self.ENC_DEPTH):
            dec.append(nn.Sequential(nn.Conv2d(ch + ch // 2, ch // 2, 3, padding=1),
                                     nn.InstanceNorm2d(ch // 2, affine=True), nn.ReLU()))
            ch //= 2
        self.dec = nn.ModuleList(dec)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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


def main() -> int:
    """パラメータ数を確認するための簡易 CLI。"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--n-ffc", type=int, default=4)
    a = ap.parse_args()

    model = StageF(base=a.base, n_ffc=a.n_ffc)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"StageF(base={a.base}, n_ffc={a.n_ffc}) params: {n_par:,}  "
          f"(全解像度で動作、1/{2 ** StageF.ENC_DEPTH} まで落として戻す)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
