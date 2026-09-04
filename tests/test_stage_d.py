"""Stage D の回帰テスト。

実装中に踏んだバグを固定する。いずれも「動くが何もしない/悪化する」種類で、
指標を見ていなければ気づけなかったもの。

  1. fill をマスク外で元画像に戻すと、勾配ドメイン合成の境界条件が恒等的に 0 になり
     調和膜が消える (何も起きない)
  2. 色統計整合をマスク内だけに掛けると fill が境界で不連続になり、
     その不連続を勾配ドメイン合成が拾って継ぎ目を悪化させる
  3. 粒状 σ をアニュラス全体の高周波から取るとシーンのテクスチャを拾って過大評価になり、
     入れすぎて継ぎ目が目立つ
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import boundary_discontinuity, grain_ratio     # noqa: E402
from nagi_inpaint2 import stage_d                                 # noqa: E402


def square_mask(n=96, lo=32, hi=64):
    m = np.zeros((n, n), dtype=bool)
    m[lo:hi, lo:hi] = True
    return m


def textured(n=96, seed=0):
    """低周波の構造 + 高周波ノイズ。実写タイルの代用。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n] / n
    base = 0.5 + 0.2 * np.sin(6 * yy) + 0.15 * np.cos(5 * xx)
    img = base[..., None] + rng.normal(0, 0.02, (n, n, 3))
    return img.astype(np.float32)


# --- 調和膜ソルバ ---------------------------------------------------------

def test_membrane_constant_boundary():
    """境界が一定なら内部も同じ定数になる (調和関数の最大値原理)。"""
    m = square_mask()
    c = stage_d._membrane(np.ones(m.shape), m)
    assert np.allclose(c[m], 1.0, atol=1e-6)


def test_membrane_linear_boundary():
    """線形な境界条件なら内部も同じ線形関数になる。"""
    m = square_mask()
    _, xx = np.mgrid[:96, :96]
    lin = xx / 95.0
    c = stage_d._membrane(lin, m)
    assert np.abs(c[m] - lin[m]).max() < 1e-6


def test_membrane_zero_boundary():
    m = square_mask()
    c = stage_d._membrane(np.zeros(m.shape), m)
    assert np.abs(c[m]).max() < 1e-9


# --- バグ1: fill をマスク外で潰すと勾配ドメインが無効化される --------------

def test_poisson_needs_fill_outside_mask():
    img = textured()
    m = square_mask()
    fill = img + 0.1                       # 全面にオフセット (マスク外にも伸びている)

    good = stage_d.poisson_blend(fill, img, m)
    assert not np.allclose(good[m], fill[m]), "膜が効いていない"

    # マスク外を元画像に戻した fill を渡すと、境界の食い違いが 0 になり膜が消える
    clipped = np.where(m[..., None], fill, img)
    bad = stage_d.poisson_blend(clipped, img, m)
    assert np.allclose(bad[m], clipped[m], atol=1e-6)


# --- バグ2: 色整合をマスク内だけに掛けると継ぎ目が悪化する ------------------

def test_color_match_applies_outside_mask():
    """契約: match_color は fill 全体に補正を掛ける (境界の連続性を保つ)。"""
    img = textured()
    m = square_mask()
    ann = stage_d.annulus_of(m)
    fill = img * 0.8 + 0.05
    out = stage_d.match_color(fill, img, m, ann)
    outside = ~stage_d.dilate(m, 1)
    assert not np.allclose(out[outside], fill[outside]), \
        "マスク外にも補正が掛かっていない (境界が不連続になる)"


# --- バグ3: 粒状 σ の過大評価 ---------------------------------------------

def test_grain_sigma_ignores_texture():
    """テクスチャの強い領域でも、σ 推定はノイズ水準に留まること。"""
    n = 128
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:n, :n]
    noise_sd = 0.01
    # 右半分だけ強い高周波テクスチャを載せる
    gray = rng.normal(0, noise_sd, (n, n))
    gray[:, n // 2:] += 0.3 * np.sin(xx[:, n // 2:] * 2.0)
    ann = np.ones((n, n), dtype=bool)

    est = stage_d.estimate_grain_sigma(gray, ann)
    naive = float(stage_d._highpass(gray)[ann].std())
    assert est < naive * 0.5, "テクスチャを拾って過大評価している"
    assert 0.3 * noise_sd < est < 3.0 * noise_sd


# --- 統合: 指標が正しい向きに動く -----------------------------------------

def test_harmonize_improves_boundary_and_grain():
    img = textured(seed=1)
    m = square_mask()
    # ぼかし + 色ずれ = 素朴なインペインタの出力を模す
    k = np.ones(9) / 9
    fill = img.copy()
    for ax in (0, 1):
        fill = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), ax, fill)
    fill = fill * 0.95 + 0.04

    b0 = boundary_discontinuity(np.where(m[..., None], fill, img), m)
    g0 = grain_ratio(np.where(m[..., None], fill, img), m)

    out = stage_d.harmonize(fill, img, m, rng=np.random.default_rng(0))
    b1 = boundary_discontinuity(out, m)
    g1 = grain_ratio(out, m)

    assert abs(b1 - 1.0) < abs(b0 - 1.0), f"継ぎ目が改善していない {b0:.2f}->{b1:.2f}"
    assert abs(g1 - 1.0) < abs(g0 - 1.0), f"粒状が改善していない {g0:.2f}->{g1:.2f}"
    # マスク外は元画像のまま
    assert np.allclose(out[~m], img[~m], atol=1e-6)


@pytest.mark.parametrize("area", [0.002, 0.02, 0.08])
def test_harmonize_various_mask_sizes(area):
    """マスク面積を変えても落ちないこと (§6.5 の 0.1%-8%)。"""
    n = 128
    img = textured(n, seed=2)
    side = int(round((area * n * n) ** 0.5))
    m = np.zeros((n, n), dtype=bool)
    c = n // 2
    m[c - side // 2:c + side // 2 + 1, c - side // 2:c + side // 2 + 1] = True
    if m.sum() == 0:
        pytest.skip("マスクが小さすぎる")
    out = stage_d.harmonize(img * 0.9, img, m, rng=np.random.default_rng(0))
    assert np.isfinite(out).all()
    assert np.allclose(out[~m], img[~m], atol=1e-6)
