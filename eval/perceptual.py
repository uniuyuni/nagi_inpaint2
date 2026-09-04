"""知覚指標 (LPIPS 相当) — 学習済み CNN の特徴空間での距離 (SPEC 目標変更に伴う追加)。

**なぜ追加するか**

目標が「PSNR で LaMa に勝つ」から「肉眼で LaMa に勝つ」へ変わった。実測で
PSNR がこの用途の品質を代表しないことが分かっている: 候補を平均化するほど
PSNR は上がるのに粒状 (grain) は失われ、写真としては劣化した (1.145 -> 0.505)。
画素値の差ではなく、CNN の特徴空間での距離のほうが「見た目の近さ」に近いことが
知られている (LPIPS, Zhang et al. 2018)。これを主指標にする。

**真の LPIPS ではないことに注意**

本家 LPIPS は VGG (や AlexNet) の各層出力に対して学習済みの 1x1 conv 線形重み
(`lin` 層) を掛けてから層を合算する。この線形重みは知覚実験データ (BAPPS) で
別途学習されたものであり、本リポジトリには無い (ネットワーク越しに取得もしない
方針)。したがって本実装は **線形重みの代わりに全チャンネル・全層を等重みで
平均する「LPIPS 相当の特徴距離」** であり、真の LPIPS 値とは一致しない。
相対比較 (同じ実装同士でどちらが近いか) の用途に限って使うこと。

**モデルの入手について**

`~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth` が既に存在するため、
`torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)` はこのキャッシュを
使い、ネットワークへは一切アクセスしない (torchvision はハッシュ付きファイル名が
キャッシュに存在すれば再ダウンロードしない)。`lpips` パッケージは未インストールで
あり、pip install もしない方針なので使わない。

**マスク付き距離について**

穴埋め評価では画像の大部分が元画像と同一 (マスク外はそもそも変更していない) なので、
画像全体で空間平均すると差が薄まって指標が鈍る。`mask` を渡した場合は、各層の
特徴マップの解像度へ `F.interpolate(..., mode="area")` でマスクを縮小し、
その重みでの加重平均を取ることで、マスク領域だけの知覚距離を測る。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# ImageNet 学習済みモデルの前提とする正規化定数。
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# VGG19 の `features` (Sequential) 内で、各ブロック最後の ReLU 直後のインデックス。
# relu1_2, relu2_2, relu3_4, relu4_4, relu5_4 相当 (LPIPS/知覚損失で定番の抽出点)。
# torchvision.models.vgg19().features を実際に列挙して確認済み:
#   3  = block1 (64ch)  の2つ目の ReLU
#   8  = block2 (128ch) の2つ目の ReLU
#   17 = block3 (256ch) の4つ目の ReLU
#   26 = block4 (512ch) の4つ目の ReLU
#   35 = block5 (512ch) の4つ目の ReLU
_VGG19_RELU_LAYERS = (3, 8, 17, 26, 35)


class Perceptual:
    """LPIPS 相当の知覚距離。**値は小さいほど良い (0 が完全一致)。**

    真の LPIPS (学習済み線形重み `lin` を層ごとに掛けて合算) ではなく、
    線形重みの代わりに全チャンネル・全層を等重みで平均した近似値である
    (モジュール docstring 参照)。
    """

    def __init__(self, net: str = "vgg19", device: str = "mps", layers=None):
        if net != "vgg19":
            # alexnet-owt-7be5be79.pth / resnet50-11ad3fa6.pth もキャッシュには
            # あるが、本タスクで要求されているのは VGG19 のみ。安易に他ネットへ
            # 分岐すると重みの正規化やレイヤ選定が network ごとに変わり検証されない
            # まま混入するため、明示的に未対応とする。
            raise NotImplementedError(
                f"net={net!r} は未対応 (vgg19 のみ実装済み)")
        self.device = device
        self.layers = tuple(layers) if layers is not None else _VGG19_RELU_LAYERS
        self._max_layer = max(self.layers)

        # weights= に enum を渡す形が、キャッシュ済みファイル名 (ハッシュ込み) と
        # 一致する限りネットワークアクセスを起こさない書き方。
        weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        model = torchvision.models.vgg19(weights=weights)
        self.features = model.features.to(device).eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self._mean = mean.to(device)
        self._std = std.to(device)

    def _prep(self, x: np.ndarray) -> torch.Tensor:
        """知覚空間 (このリポジトリの慣習で値域はおおよそ 0..1.2) を、
        ImageNet 正規化済み (1,3,H,W) tensor に変換する。

        `eval/lama_failures.py` の `to_u8` と同じ流儀で 0..1.2 -> 0..1 に丸める。
        """
        x = np.clip(x / 1.2, 0.0, 1.0).astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        t = t.to(self.device)
        return (t - self._mean) / self._std

    def _extract(self, t: torch.Tensor) -> list[torch.Tensor]:
        """要求した層 (ReLU 直後) の出力だけを集めて返す。"""
        want = set(self.layers)
        feats = []
        h = t
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in want:
                feats.append(h)
            if i >= self._max_layer:
                break
        return feats

    def __call__(self, a: np.ndarray, b: np.ndarray,
                 mask: np.ndarray | None = None) -> float:
        """`a` と `b` の知覚距離 (float, 小さいほど良い)。

        `mask` (H,W, 0/1 または bool) を渡すと、そのマスク領域だけを対象にした
        加重平均に切り替わる (モジュール docstring 「マスク付き距離について」参照)。
        """
        with torch.inference_mode():
            fa = self._extract(self._prep(a))
            fb = self._extract(self._prep(b))

            mt = None
            if mask is not None:
                mt = torch.from_numpy(
                    np.ascontiguousarray(mask.astype(np.float32)))[None, None]
                mt = mt.to(self.device)

            layer_dists = []
            for xa, xb in zip(fa, fb):
                # チャンネル方向に L2 正規化してから差の二乗を取る (LPIPS の標準手順)。
                na = xa / (xa.norm(p=2, dim=1, keepdim=True) + 1e-10)
                nb = xb / (xb.norm(p=2, dim=1, keepdim=True) + 1e-10)
                d = (na - nb) ** 2
                # 学習済み線形重みが無いので、チャンネルは等重みで平均する
                # (モジュール docstring 参照)。結果は (1,1,h,w) の空間マップ。
                d = d.mean(dim=1, keepdim=True)

                if mt is not None:
                    m = F.interpolate(mt, size=d.shape[-2:], mode="area")
                    val = (d * m).sum() / (m.sum() + 1e-8)
                else:
                    val = d.mean()
                layer_dists.append(val)

            return float(torch.stack(layer_dists).mean().item())
