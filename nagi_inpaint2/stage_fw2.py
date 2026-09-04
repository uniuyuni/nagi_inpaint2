"""StageFW2 — デコーダを K 本に分岐させた勝者総取り (WTA) モデル。

**効く理屈と、効かない条件 (このファイルの存在理由)**

狙い: L1 損失は多峰性の分布に対して平均を出すためぼける (実測: grain 0.109、
LaMa 0.943、理想 1.0)。K 個の仮説を出して**最良の1つにだけ勾配を流す**
(Winner-Take-All) と、平均を作る動機が消える。GAN と違い識別器も min-max も
無いので学習は安定する (GAN は2回試して崩壊した)。

**効かない条件**: 分岐が浅いと再び同一の仮説に収束する。`StageFW`
(`nagi_inpaint2/stage_fw.py`) は**出力層 (最後の 3x3 畳み込み) だけ**を K 個に
した。4つのヘッドはバックボーン全体を共有し、**同じ特徴の線形読み出しの
違いしか表現できない**。結果、仮説がほぼ同一になり、**正解を見て最良を選ぶ
神託選択ですら first 選択を上回らなかった** (PSNR 21.29 vs 21.43)。仮説間の
損失の標準偏差も損失の約10%しかなかった。

**検出方法**: 「神託選択が first 選択を明確に上回るか」。上回らなければ
多様性が無く、WTA は機能していない。`train/train_stage_fw2.py` はこの差
(`oracle - first`) を毎回の検証ログに出す。

**このファイルの対処**: `StageF` (`nagi_inpaint2/stage_f.py`) の構造は
stem -> エンコーダ2段 (全解像度の 1/4 まで) -> FFC ブロック (文脈の理解、
1/4 解像度) -> デコーダ2段 (全解像度に復元) -> 出力層、という並び。
「文脈の理解」(エンコーダ+FFC) は仮説によらず共通でよいはずだが、「どう
埋めるか」を決めるデコーダは仮説ごとに違う関数であるべきなので、**デコーダ
以降 (dec 2段 + out) を K 本の完全に独立した重みの枝にする**。さらに各枝の
入力 (FFC ブロックの出力) に、枝ごとの学習可能な埋め込み `emb[k]`
(形 (1, C, 1, 1)) を加算する。理由: 枝の重みだけだと初期値の差が小さく、
同じ解に収束しやすい。明示的な「枝の識別子」を与えると分岐が保たれやすい。

**実装方針**: `StageF` をコピーせず import して再利用する。共有部分
(stem/エンコーダ/FFC ブロック) は `StageF` のインスタンスから同じ構造の
モジュールをそのまま流用し (`load_backbone` で checkpoint のキー名がそのまま
一致するように)、デコーダ以降だけを `_DecoderBranch` として K 個複製する。

**パラメータ数**: base=96, k=4, n_ffc=4 で約 11.7M (20M 以内)。共有部分
(stem+enc+mid) が約 6.74M、分岐1本 (dec+out) が約 1.25M × 4 ≈ 4.99M、
emb が K × C ≈ 無視できる量。デコーダ2段とも分岐させても 20M を超えないため、
分岐段数を減らす調整は不要だった (超える場合は dec の後段だけ分岐する設計に
縮小する想定だったが、今回は全段で収まった)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nagi_inpaint2.stage_f import StageF

_EMB_INIT_STD = 0.02   # 枝の識別子 (emb) の初期化標準偏差。仕様: N(0, 0.02^2)
# 枝の重みに加える相対摂動。**初期の多様性を明示的に作るための値。**
# 実測: emb (std=0.02) だけでは事前学習重みからの初期化後の仮説間差が
# 4.4e-8 と実質ゼロで、出力層のみ分岐する StageFW (4.4e-4) より 1万倍小さい。
# その StageFW ですら学習後に多様性が生まれず WTA が機能しなかったので、
# 重みそのものを相対的に揺らして対称性を壊す。
_BRANCH_PERTURB = 0.08


class _DecoderBranch(nn.Module):
    """`StageF` のデコーダ (dec 2段) + 出力層 (out) と全く同じ構造を持つ、
    1本の独立した枝。

    `StageF.forward` のデコーダループと同じ計算 (skip を使った nearest
    アップサンプル + 畳み込み) を行うが、skip のリストを**破壊的に pop
    しない** (K 本の枝が同じ skip リストを読み取り専用で共有するため)。
    """

    def __init__(self, base: int, enc_depth: int):
        super().__init__()
        ch = base * (2 ** enc_depth)
        dec = []
        for _ in range(enc_depth):
            dec.append(nn.Sequential(nn.Conv2d(ch + ch // 2, ch // 2, 3, padding=1),
                                     nn.InstanceNorm2d(ch // 2, affine=True), nn.ReLU()))
            ch //= 2
        self.dec = nn.ModuleList(dec)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)
        self._init_weights()

    def _init_weights(self) -> None:
        """枝ごとの乱数初期化を ReLU 向け Kaiming 分布でやり直す。

        **なぜ**: `nn.Conv2d` の既定初期化 (`kaiming_uniform_(a=sqrt(5))`) は
        全結合層にも使える汎用の控えめな設定で、実際に使う非線形性 (ReLU) に
        対して分散が小さすぎる。各段が InstanceNorm2d を挟むため (ここは
        `StageF` の checkpoint をそのまま読めるようにするために外せない)、
        分岐直後の弱い乱数は 1〜2 段の IN で平均・分散が正規化されてほぼ
        打ち消され、K 本の枝が実質同じ出力に収束してしまうことを実測で確認
        した (`base=16/96` で `StageFW` (出力層のみ分岐) との仮説間差の比が
        既定初期化では 0.65〜1.2 倍 (時に StageFW 未満) だったのが、ReLU 向け
        Kaiming 分布に変えるだけで 4〜7 倍まで伸びた)。**IN で消えない量まで
        枝ごとの乱数を強めに振る**ことが、独立した重みだけで多様性を作るのに
        必要だった。load_backbone を呼べばこの乱数はそのまま上書きされるので
        本学習には影響しない (影響するのは --init 無し / テストのみ)。
        """
        for layer in self.dec:
            conv = layer[0]
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="relu")
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, dec_skips: list[torch.Tensor]) -> torch.Tensor:
        """x: FFC ブロック出力 (+ emb 加算済み)。dec_skips: 浅い順ではなく
        `StageF.forward` と同じ消費順 (深い解像度側から)。"""
        for d, skip in zip(self.dec, dec_skips):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = d(torch.cat([x, skip], 1))
        return self.out(x)


class StageFW2(nn.Module):
    """`StageF` と同じ共有部分 (stem+エンコーダ+FFC) を持ち、デコーダ以降を
    K 本の独立した枝にした WTA モデル。

    forward の出力は (N, K, 3, H, W)。`StageF`/`StageFW` と同じく呼び出し側が
    マスク合成を行う前提でマスク処理はしない。
    """

    def __init__(self, base: int = 96, k: int = 4, **kw):
        super().__init__()
        self.k = k
        self.base = base
        # 共有部分は StageF と全く同じモジュールをそのまま使う。checkpoint の
        # キー名 ("stem.*", "enc.*", "mid.*") が一致するようにするため、
        # StageF を一度構築してその該当サブモジュールを借用する
        # (dec/out は使わないので破棄する)。
        ref = StageF(base=base, **kw)
        self.stem = ref.stem
        self.enc = ref.enc
        self.mid = ref.mid
        self.enc_depth = StageF.ENC_DEPTH
        mid_ch = base * (2 ** self.enc_depth)
        self.mid_ch = mid_ch

        self.branches = nn.ModuleList(
            [_DecoderBranch(base, self.enc_depth) for _ in range(k)])
        # 枝ごとの学習可能な埋め込み (形 (1,C,1,1))。枝の重みだけだと初期値の差が
        # 小さく同じ解に収束しやすいため、明示的な「枝の識別子」を与える。
        self.emb = nn.ParameterList(
            [nn.Parameter(torch.empty(1, mid_ch, 1, 1)) for _ in range(k)])
        for e in self.emb:
            nn.init.normal_(e, mean=0.0, std=_EMB_INIT_STD)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img * (1 - mask), mask], 1)
        x = self.stem(x)
        skips = [x]
        for e in self.enc:
            x = e(x)
            skips.append(x)
        mid_out = self.mid(x)                      # (N, mid_ch, H/4, W/4)

        # StageF.forward のデコーダループが消費する順序 (skips[-2] を毎回取り、
        # 末尾を pop する) を、破壊的操作なしで再現する: skips[:-1] を逆順にした
        # リスト (深い解像度側から浅い側へ) を、pop せず読み取るだけにする。
        dec_skips = list(reversed(skips[:-1]))

        outs = []
        for bi in range(self.k):
            xb = mid_out + self.emb[bi]
            outs.append(self.branches[bi](xb, dec_skips))
        return torch.stack(outs, dim=1)             # (N, K, 3, H, W)

    def load_backbone(self, state_dict: dict) -> None:
        """`StageF` の state_dict から共有部分 (stem/enc/mid) を読み込み、
        **K 本の枝すべてを同じ事前学習済みデコーダ重み (dec+out) で初期化**する。

        **なぜ K 本とも同じデコーダ重みで初期化するか**: 学習時間を節約する
        ため (`StageFW` と同じ理由)。重みが同一でも、各枝の入力に加わる
        `emb[k]` が枝ごとに異なる (このファイル冒頭 docstring 参照) ので、
        `StageFW` の出力層のみの分岐と違って対称性は壊れている。

        `emb` は N(0, 0.02^2) で再初期化する (`__init__` の初期化と同じ分布を
        明示的に保証するため)。
        """
        shared_sd, dec_sd, out_sd = {}, {}, {}
        for key, v in state_dict.items():
            if key.startswith("stem.") or key.startswith("enc.") or key.startswith("mid."):
                shared_sd[key] = v
            elif key.startswith("dec."):
                dec_sd[key[len("dec."):]] = v
            elif key.startswith("out."):
                out_sd[key[len("out."):]] = v
            else:
                raise ValueError(f"想定外のキー (StageF の state_dict ではない?): {key}")

        missing, unexpected = self.load_state_dict(shared_sd, strict=False)
        assert not unexpected, f"想定外のキー: {unexpected}"
        # missing には分岐 (branches.*/emb.*) が必ず含まれる。共有部分 3 種
        # (stem/enc/mid) が shared_sd に全て含まれていたことだけ確認する。
        loaded_prefixes = {k_.split(".")[0] for k_ in shared_sd}
        assert loaded_prefixes <= {"stem", "enc", "mid"}, \
            f"共有部分以外のキーが混入している: {loaded_prefixes}"

        for branch in self.branches:
            m, u = branch.dec.load_state_dict(dec_sd, strict=True)
            branch.out.load_state_dict(out_sd, strict=True)

        # **枝の重みに相対摂動を加えて対称性を壊す** (_BRANCH_PERTURB 参照)。
        # 枝0 は事前学習済みのまま残す (退行しないための保険)。
        with torch.no_grad():
            for j, branch in enumerate(self.branches):
                if j == 0:
                    continue
                for prm in list(branch.dec.parameters()) + list(branch.out.parameters()):
                    prm.mul_(1.0 + _BRANCH_PERTURB * torch.randn_like(prm))
            for e in self.emb:
                e.normal_(mean=0.0, std=_EMB_INIT_STD)


def load_stage_fw2(path: str, device: str):
    """checkpoint から `StageFW2` を復元する (`nagi_inpaint2.stage_fw.load_stage_fw`
    と同じ役割の StageFW2 版)。

    `base`/`k` は checkpoint のメタデータから復元する。`StageFW2` は
    `StageFW` と違って出力層の形から base/k を逆算できない (デコーダが枝ごとに
    別の重みを持つため単純な shape 照合ができない) ので、`train_stage_fw2.py`
    が checkpoint に必ず `base`/`k` を保存する前提とする。
    """
    ck = torch.load(path, map_location=device)
    base = int(ck.get("base") or 0)
    k = int(ck.get("k") or 0)
    if not base or not k:
        raise ValueError(f"base/k を checkpoint から復元できない (メタデータ欠落): {path}")
    model = StageFW2(base=base, k=k).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def main() -> int:
    """パラメータ数を確認するための簡易 CLI (`StageF`/`StageFW` と同じ流儀)。"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-ffc", type=int, default=4)
    a = ap.parse_args()

    model = StageFW2(base=a.base, k=a.k, n_ffc=a.n_ffc)
    n_par = sum(q.numel() for q in model.parameters())
    shared = sum(q.numel() for q in model.stem.parameters()) \
        + sum(q.numel() for q in model.enc.parameters()) \
        + sum(q.numel() for q in model.mid.parameters())
    per_branch = sum(q.numel() for q in model.branches[0].parameters())
    print(f"StageFW2(base={a.base}, k={a.k}, n_ffc={a.n_ffc}) params: {n_par:,}  "
          f"(共有 {shared:,} + 分岐1本 {per_branch:,} x {a.k} + emb "
          f"{sum(q.numel() for q in model.emb.parameters()):,})  "
          f"{'OK (<=20M)' if n_par <= 20_000_000 else '**20M 超過**'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
