"""ブートストラップ再探索 (`inpaint(passes=...)`) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない)。

**なぜ multi-pass が要るか**: `solve_blocks` の照合コストは `query` (Stage A の
低周波補完、ぼけている) の特徴だけを見るので、高周波テクスチャで候補を判別
できない。1パス目の出力 `paste(...)` は実画素のコピーなので高周波テクスチャを
持つ。これを2パス目のクエリに差し替えれば、以降のパスは高周波を含むコストで
候補を選び直せる (古典的なマルチパス・テクスチャ合成)。**平均化はしない** —
中間パスも `paste` (1ブロック1候補のコピー) を通すだけで、テクスチャをぼかす
操作は挟まらない。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b_search import (   # noqa: E402
    inpaint, lowfreq_query, paste, solve_blocks,
)

DEVICE = "cpu"


def _periodic_texture(n: int, noise: float, seed: int) -> torch.Tensor:
    """x 方向にのみ周期的な (= y 方向には不変な) テクスチャ (test_topk.py と同じ構成)。"""
    yy, xx = np.mgrid[:n, :n]
    base = 0.5 + 0.4 * np.sin(xx / 6.0)
    rng = np.random.default_rng(seed)
    img_np = (np.repeat(base[..., None], 3, axis=2)
             + rng.normal(0, noise, (n, n, 3))).astype(np.float32)
    return torch.from_numpy(img_np.transpose(2, 0, 1)[None]).to(DEVICE)


def _center_mask(n: int, lo: int, hi: int) -> torch.Tensor:
    m = np.zeros((n, n), dtype=np.float32)
    m[lo:hi, lo:hi] = 1.0
    return torch.from_numpy(m[None, None]).to(DEVICE)


def _blur_query(img: torch.Tensor, k: int = 9) -> torch.Tensor:
    """`img` をぼかして query を作る (Stage A の低周波補完の代用)。"""
    r = k // 2
    return F.avg_pool2d(F.pad(img, (r, r, r, r), mode="replicate"), k, stride=1)


# --- テスト1: 既定経路の不変性 (passes 未指定 == passes=1) --------------------

def test_passes_default_matches_explicit_passes1():
    img = _periodic_texture(64, noise=0.02, seed=3)
    mask = _center_mask(64, 18, 46)
    q = _blur_query(img)

    out_default = inpaint(img, mask, query=q)
    out_p1 = inpaint(img, mask, query=q, passes=1)
    assert torch.equal(out_default, out_p1)


# --- テスト2: query が無ければ passes は無視される ----------------------------

def test_passes_ignored_without_query():
    img = _periodic_texture(56, noise=0.02, seed=5)
    mask = _center_mask(56, 20, 36)

    out_no_passes = inpaint(img, mask)
    out_p3 = inpaint(img, mask, passes=3)
    assert torch.equal(out_no_passes, out_p3)


# --- テスト3: 中間クエリはマスク外を元画像に戻すが、最終出力は戻さない ---------

def test_intermediate_query_resets_outside_mask_but_final_output_extends():
    """`passes=2` の中間クエリ生成が正しいことを、`inpaint` の内部手順を手で
    再現して確認する。

    `paste` はマスク外にも合成結果を伸ばす契約 (SPEC §4.4) を持つ。この
    ブロック実装ではマスク外のブロックはコストが全チャンネルで 0 に潰れる
    ため (`c * mask` でマスク外は必ず 0)、argmin は決定的に最初のオフセット
    (`offs[0]`, ゼロでない) を選び、マスク外の画素も一律シフトされる —
    つまり素の `paste` の出力はマスク外で `img` と一致しない。

    - 中間パス (次のクエリを作る際) はこのマスク外のシフトを **必ず元画像へ
      戻す** (`torch.where(mask > 0.5, out_mid, img)`)。戻さないと誤差が
      パスをまたいで累積する。
    - 最終パスは既存の分岐 (ここでは単純に `paste`) をそのまま返すので、
      マスク外は戻らない (`paste` の契約どおり合成のまま伸びる)。

    この非対称 ("中間は戻す、最終は戻さない") を明示的に確認する。
    """
    img = _periodic_texture(64, noise=0.02, seed=3)
    mask = _center_mask(64, 18, 46)
    q = _blur_query(img)
    outside = mask.expand_as(img) <= 0.5

    # inpaint(passes=2) の内部手順 (1パス目 solve+paste -> クエリ差し替え) を
    # 手で再現する。
    idx_mid, offs_mid, _ = solve_blocks(img, mask, query=q)
    out_mid = paste(img, mask, idx_mid, offs_mid)
    q2 = torch.where(mask > 0.5, out_mid, img)

    # 対比: 戻す前の生 paste 結果はマスク外で img と一致しない
    # (paste がマスク外にも合成を伸ばす契約を実際に発動させていることの確認)。
    assert not torch.equal(out_mid[outside], img[outside])
    # 中間クエリはマスク外で元画像に戻っていること。
    assert torch.equal(q2[outside], img[outside])

    # この中間クエリで最終パスを解いた結果が、inpaint(passes=2) の実際の
    # 出力と bit 一致することを確認する (中間クエリ生成の正しさの間接検証)。
    idx_final, offs_final, _ = solve_blocks(img, mask, query=q2)
    out_final_expected = paste(img, mask, idx_final, offs_final)
    out_actual = inpaint(img, mask, query=q, passes=2)
    assert torch.equal(out_actual, out_final_expected)

    # 最終出力はマスク外を戻さない (paste の契約どおり合成のまま伸びている)。
    assert not torch.equal(out_actual[outside], img[outside])


# --- テスト4: 収束性 (passes=2 は passes=1 以下の MSE) -------------------------

def test_passes2_reduces_mse_vs_gt():
    """低周波 query では高周波テクスチャの位相を判別できないことを利用して、
    passes=1 が誤ったオフセットを選びやすい状況を作り、passes=2 (ブートスト
    ラップ後の高周波クエリ) がそれを補正して GT に近づくことを確認する。

    GT は高周波の周期テクスチャ (周期 8px)。`lowfreq_query` の縮小率 8 で
    ぼかすと、8px 周期はちょうど打ち消されてクエリからほぼ消える
    (面積平均が整数周期分になるため)。1パス目はこの「ほぼ情報が無い」
    クエリで候補を選ぶしかないので、真の位相とズレたオフセットを選びやすい。
    2パス目は 1パス目が実際にコピーした (高周波を含む) 画素を新しいクエリに
    使うので、真の位相との一致度で候補を選び直せる。

    パラメータ (周期・振幅・ノイズ・縮小率・シード・探索半径) はこの非等式
    が安定して成り立つ値を実測して選んだもの。シードは固定 (乱数の揺れなし)。
    """
    n, lo, hi = 96, 32, 64
    fine_period, fine_amp, noise, seed, shrink = 8.0, 0.3, 0.01, 1, 8

    yy, xx = np.mgrid[:n, :n]
    fine = fine_amp * np.sin(2 * np.pi * xx / fine_period)
    gt_np = np.repeat((0.5 + fine)[..., None], 3, axis=2).astype(np.float32)
    rng = np.random.default_rng(seed)
    noisy_np = gt_np + rng.normal(0, noise, (n, n, 3)).astype(np.float32)

    img = torch.from_numpy(noisy_np.transpose(2, 0, 1)[None]).to(DEVICE)
    gt = torch.from_numpy(gt_np.transpose(2, 0, 1)[None]).to(DEVICE)
    mask = _center_mask(n, lo, hi)
    # マスク外は真の画素、マスク内だけ低周波にぼかす (lowfreq_query の契約)。
    q = lowfreq_query(img, mask, img, shrink=shrink)

    out1 = inpaint(img, mask, query=q, passes=1, radius=64, stride=4)
    out2 = inpaint(img, mask, query=q, passes=2, radius=64, stride=4)

    m = mask.expand_as(img) > 0.5
    mse1 = ((out1 - gt)[m] ** 2).mean().item()
    mse2 = ((out2 - gt)[m] ** 2).mean().item()
    assert mse2 <= mse1


# --- テスト5: パッチ径の露出 --------------------------------------------------

def test_patch_size_is_exposed_and_runs():
    img = _periodic_texture(56, noise=0.02, seed=5)
    mask = _center_mask(56, 20, 36)
    q = _blur_query(img)

    out = inpaint(img, mask, query=q, patch=11)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()
