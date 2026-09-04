"""StageFW — 勝者総取り (Winner-Take-All) 学習で「ぼけない補完」を狙う検証モデル。

**背景 (なぜ WTA が要るか)**

`nagi_inpaint2/stage_f.py` の `StageF` は L1 損失で学習しており、平均値回帰で
ぼける。実測: 出力の grain (マスク内外の σ 比、理想 1.0) が **0.109**。
LaMa は 0.943。人物マスクの出力を目視すると、穴の位置にぼけた塊が残り、
明らかに破綻している。

敵対的損失 (GAN) で直そうとして2回失敗した (grain 0.322 → 0.339 と動かず、
長く回すと崩壊して勝率 34/60 → 10/60)。batch 4・M1 では GAN を安定させられ
なかった。

L1 がぼけるのは「多峰性の分布に対して平均を出す」から。WTA は K 個の仮説を
出し、**最良の1つにだけ勾配を流す**ので、平均を作る動機が消える。学習は L1
と同程度に安定で、識別器も min-max も無い。

**実装方針**: `StageF` をコピーせず import して再利用する。最終出力層だけを
`3*K` チャンネルにすれば、エンコーダ/デコーダ (FFC ブロックを含む) はそのまま
使い回せる。`StageF` の checkpoint から backbone を読み込めるようにして
(`load_backbone`)、学習時間を節約する。出力層だけは形が変わる (3ch -> 3*Kch)
ので読み込めない。K 個とも同じ重みで初期化し、ごく小さな乱数 (std=1e-3) を
加えて対称性を破る。完全に同じ重みのままだと全ヘッドが同じ答えを出し続け、
WTA が機能しない (勝者が常に「たまたま先に並んだヘッド」に固定されてしまう)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nagi_inpaint2.stage_f import StageF

_OUT_INIT_NOISE_STD = 1e-3   # 出力層の対称性を破るための小さな乱数の標準偏差


class StageFW(nn.Module):
    """`StageF` と同じ構造で、出力を K 個の仮説にする。

    forward の出力は (N, K, 3, H, W)。呼び出し側 (損失関数や推論コード) が
    仮説の中から1つを選ぶ、または WTA 損失で全てを使う。`StageF` 自身と同じく
    マスク処理はしない (マスク外もネットワークの生出力)。
    """

    def __init__(self, base: int = 96, k: int = 4, **kw):
        super().__init__()
        self.k = k
        self.backbone = StageF(base=base, **kw)
        in_ch = self.backbone.out.in_channels
        # 出力層だけ 3ch -> 3*Kch に差し替える。エンコーダ/デコーダ/FFC はそのまま。
        self.backbone.out = nn.Conv2d(in_ch, 3 * k, 3, padding=1)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.backbone(img, mask)                 # (N, 3*K, H, W)
        n, _, h, w = x.shape
        return x.view(n, self.k, 3, h, w)

    def load_backbone(self, state_dict: dict) -> None:
        """`StageF` の state_dict から出力層以外を読み込む。

        **なぜ**: 学習時間を節約するため (背景参照)。エンコーダ/デコーダ/FFC は
        `StageF` と全く同じ形なのでそのまま読み込めるが、出力層は 3ch から
        3*Kch に形が変わるので読み込めない。出力層は K 個とも同じ重みで初期化
        し、ごく小さな乱数 (std=1e-3) を加えて対称性を破る。加えないと全ヘッド
        が同じ答えを出し続け、WTA が機能しない (常に同じヘッドが「たまたま」
        勝ち続け、実質 L1 と同じになる)。
        """
        backbone_sd = {k_: v for k_, v in state_dict.items()
                       if not k_.startswith("out.")}
        missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
        assert not unexpected, f"想定外のキー: {unexpected}"
        assert set(missing) <= {"out.weight", "out.bias"}, \
            f"出力層以外にも読み込めなかったキーがある: {missing}"

        ow, ob = state_dict["out.weight"], state_dict["out.bias"]   # (3,C,3,3), (3,)
        with torch.no_grad():
            w = ow.repeat(self.k, 1, 1, 1)                          # (3*K,C,3,3)
            b = ob.repeat(self.k)                                   # (3*K,)
            w = w + torch.randn_like(w) * _OUT_INIT_NOISE_STD
            b = b + torch.randn_like(b) * _OUT_INIT_NOISE_STD
            self.backbone.out.weight.copy_(w)
            self.backbone.out.bias.copy_(b)


def select_hypothesis(preds: torch.Tensor, x: torch.Tensor, mk: torch.Tensor,
                      hyp: str, copy_out: torch.Tensor | None = None) -> torch.Tensor:
    """(N,K,3,H,W) の仮説群から1つを選び (N,3,H,W) にする (eval 専用ヘルパー)。

    `eval/benchmark.py`, `eval/compare_visual.py` の両方から使う共通ロジック。

    - "first"  : 仮説0。選択コストが要らない、想定される製品経路。
    - "oracle" : GT (`x`) に最も近い仮説を選ぶ。**上限測定専用**。GT を見るので
                 製品経路ではない。K 個の仮説の中に良いものが実際にあるか
                 (多様性が生まれているか) を測るための診断値。
    - "copy"   : Stage B のコピー結果 (`copy_out`) にマスク内で最も近い仮説を
                 選ぶ。GT を使わない、実用的な選択方法。
    """
    if hyp == "first":
        return preds[:, 0]
    m = mk.expand(-1, 3, -1, -1).unsqueeze(1)          # (N,1,3,H,W)
    if hyp == "oracle":
        ref = x.unsqueeze(1)
    elif hyp == "copy":
        if copy_out is None:
            raise ValueError("hyp='copy' には copy_out が必要")
        ref = copy_out.unsqueeze(1)
    else:
        raise ValueError(f"unknown hyp: {hyp}")
    d = ((preds - ref).abs() * m).sum(dim=(2, 3, 4))    # (N,K)
    idx = d.argmin(dim=1).view(-1, 1, 1, 1, 1).expand(-1, 1, *preds.shape[2:])
    return preds.gather(1, idx).squeeze(1)


def load_stage_fw(path: str, device: str):
    """checkpoint から `StageFW` を復元する。

    `base`/`k` は checkpoint のメタデータ (無ければ `out.weight` の形) から
    復元する。`eval/benchmark.py` の `StageF` 用ロード規約 (`base` を
    `out.weight.shape[1]` から復元する) に倣う。

    戻り値: (model, checkpoint dict)
    """
    ck = torch.load(path, map_location=device)
    # **出力層のキーは "backbone.out.weight"。** `StageFW` は `StageF` を
    # `backbone` として内包しているので、`StageF` 用の "out.weight" は存在しない。
    # メタデータの base/k を優先し、無い場合だけ形から復元する。
    ow = ck["model"].get("backbone.out.weight")
    base = int(ck.get("base") or (ow.shape[1] if ow is not None else 0))
    k = int(ck.get("k") or (ow.shape[0] // 3 if ow is not None else 0))
    if not base or not k:
        raise ValueError(f"base/k を復元できない: {path}")
    model = StageFW(base=base, k=k).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


class HypothesisQuery:
    """`StageFW` を `StageF` と同じ呼び出し規約 (`model(x, mk) -> (N,3,H,W)`) に
    揃えるアダプタ。

    `eval/benchmark.py` / `eval/compare_visual.py` は `StageF` 用に書かれた
    既存経路 (`_build_query`, `sf(x, mk)` 呼び出し) をそのまま使い回す。
    `StageFW` は (N,K,3,H,W) を返すので、このラッパーが `select_hypothesis`
    で1つに絞ってから返す。
    """

    def __init__(self, model: StageFW, hyp: str):
        self.model = model
        self.hyp = hyp

    def __call__(self, x: torch.Tensor, mk: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds = self.model(x, mk)
        copy_out = None
        if self.hyp == "copy":
            from nagi_inpaint2 import stage_b_search as sbs
            with torch.no_grad():
                copy_out = sbs.inpaint(x, mk)
        return select_hypothesis(preds, x, mk, self.hyp, copy_out=copy_out)
