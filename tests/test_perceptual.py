"""知覚指標 (LPIPS 相当, `eval/perceptual.py`) の回帰テスト。

device は "cpu" 固定 (mps は学習ジョブが使用中のため使わない)。

**この指標の存在理由**: PSNR は候補を平均化するほど上がる一方、粒状 (grain) は
失われて写真としては劣化する — という実測が「肉眼で LaMa に勝つ」への目標変更の
根拠になっている。したがって `Perceptual` が「ノイズが増えれば距離が増える」
「ぼけれ ば距離が増える」という単調性を持たなければ、指標として意味がない。
テスト2・3 はこの単調性そのものを確認する (特にテスト3: 我々の弱点は「ぼけ」
なので、指標がぼけに反応しなければ本末転倒)。

モデルは `~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth` の既存キャッシュ
から読む (新規ダウンロードなし)。`Perceptual` の構築コストが高いので、
モジュール scope の fixture で使い回す。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.perceptual import Perceptual   # noqa: E402

DEVICE = "cpu"
N = 128


def _photo_like(n: int, seed: int) -> np.ndarray:
    """自然画像に近い、細かいディテールと構造を両方持つ合成テクスチャ。

    低周波の構造 (sin/cos) の上に微小ノイズを重ねる。ぼかしで細部が失われ、
    かつノイズ付加で全体が変化する — という両方のテストに使える形にしている。
    値域はこのリポジトリの慣習 (知覚空間, おおよそ 0..1.2) に合わせる。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    base = (0.5
            + 0.2 * np.sin(xx / 5.0)
            + 0.2 * np.cos(yy / 7.0)
            + 0.1 * np.sin((xx + yy) / 3.0))
    img = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.2).astype(np.float32)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        out[..., c] = gaussian_filter(x[..., c], sigma=sigma)
    return out


@pytest.fixture(scope="module")
def perceptual() -> Perceptual:
    return Perceptual(device=DEVICE)


# --- 1. 同一画像の距離は 0 ---------------------------------------------------

def test_identical_images_distance_near_zero(perceptual):
    x = _photo_like(N, seed=0)
    d = perceptual(x, x)
    assert d < 1e-5


# --- 2. 単調性: ノイズが強いほど距離が大きい --------------------------------

def test_noise_monotonicity(perceptual):
    x = _photo_like(N, seed=1)
    rng = np.random.default_rng(2)
    noisy_low = np.clip(x + rng.normal(0, 0.01, x.shape).astype(np.float32), 0, 1.2)
    noisy_high = np.clip(x + rng.normal(0, 0.10, x.shape).astype(np.float32), 0, 1.2)

    d_low = perceptual(x, noisy_low)
    d_high = perceptual(x, noisy_high)

    print(f"\n[noise] sigma=0.01 -> {d_low:.6f}  sigma=0.10 -> {d_high:.6f}")
    assert d_high > d_low


# --- 3. ぼかしを検出できること (最重要) -------------------------------------

def test_blur_monotonicity(perceptual):
    """我々の弱点は「ぼけ」なので、指標がぼけに反応しなければ意味がない。"""
    x = _photo_like(N, seed=3)
    weak = _blur(x, sigma=0.8)
    strong = _blur(x, sigma=3.0)

    d_weak = perceptual(x, weak)
    d_strong = perceptual(x, strong)

    print(f"\n[blur] weak(sigma=0.8) -> {d_weak:.6f}  strong(sigma=3.0) -> {d_strong:.6f}")
    assert d_strong > d_weak


# --- 4. マスク指定が効くこと -------------------------------------------------

def test_mask_focuses_distance(perceptual):
    """画像の一部だけを変更したとき、その領域を指すマスクを渡した距離は、
    全面平均の距離より大きい (大部分が同一画像なので全面平均だと差が薄まる)。
    """
    x = _photo_like(N, seed=4)
    y = x.copy()
    lo, hi = N // 2 - 24, N // 2 + 24
    patch = _photo_like(hi - lo, seed=5)[: hi - lo, : hi - lo]
    y[lo:hi, lo:hi] = patch

    mask = np.zeros((N, N), dtype=np.float32)
    mask[lo:hi, lo:hi] = 1.0

    d_full = perceptual(x, y)
    d_masked = perceptual(x, y, mask=mask)

    print(f"\n[mask] full -> {d_full:.6f}  masked -> {d_masked:.6f}")
    assert d_masked > d_full
