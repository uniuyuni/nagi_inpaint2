"""多重解像度ブレンド (ラプラシアンピラミッド合成) の回帰テスト。

device は "cpu" 固定 (mps は学習ジョブが使用中のため使わない。ピラミッドの
ダウン/アップサンプルも `avg_pool2d`/明示サイズの `bilinear` のみを使っており
`mode="area"` を踏まないので MPS でも安全なはずだが、テストは cpu で行う)。

**背景**: 人物マスクの出力を目視で確認したところ、2つの破綻があった:
  1. 32px ブロックの格子が見える (隣接ブロックが別々の場所からコピーされ、
     境界で模様が食い違う。市松模様として肉眼で分かる)
  2. 貼ったブロックが周囲より明るく浮く (四角い明度の段差が見える)

過去に試して失敗した対処:
  - `paste_feather` (一律の羽化): 段差エネルギーは 75% 減ったが、格子が
    **にじみ**に化けただけ。1ケースは 42.3 -> 18.0 dB と壊滅した。
    全周波数を同じ幅で混ぜたのが原因。
  - `hf_transfer` (単一スケールの高周波移植): psd 0.1037 -> 0.1228 と悪化、
    boundary も 1.08 -> 1.77 と悪化した。

`band_composite` は周波数帯ごとに `base` (StageF、配置は正しいがぼけている)
と `copy` (ブロックコピー、テクスチャは正しいが配置がずれる) の出処を切り替える
ことで、この2つの失敗を避ける。このテストの主眼はテスト5 (格子が減ること) と
テスト3 (低周波が base 由来であること) — どちらも `band_composite` の存在理由
そのものなので、これらが無いテストスイートは要件を満たさない。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b_search import (   # noqa: E402
    BLOCK, _collapse, _gaussian_blur, _lap_pyr, band_composite, inpaint, paste,
    photometric_align,
)

DEVICE = "cpu"


# --- テスト1: ピラミッドの往復 -------------------------------------------------

def test_pyramid_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 96, 96).to(DEVICE)
    rec = _collapse(_lap_pyr(x, 5))
    assert torch.allclose(rec, x, atol=1e-5)


def test_pyramid_roundtrip_odd_size():
    """奇数サイズ (129x131) でも往復できること。奇数サイズは avg_pool2d で
    切り捨てられるが、upsample は常に一つ細かい層の実サイズを指定するので
    情報を失わずに戻る (`_lap_pyr`/`_collapse` の docstring 参照)。"""
    torch.manual_seed(1)
    x = torch.randn(1, 3, 129, 131).to(DEVICE)
    rec = _collapse(_lap_pyr(x, 5))
    assert rec.shape == x.shape
    assert torch.allclose(rec, x, atol=1e-5)


# --- テスト2: photometric_align が浮きを消すこと --------------------------------

def _block_means(x: torch.Tensor, block: int) -> torch.Tensor:
    return F.avg_pool2d(x, block)


def test_photometric_align_removes_offset():
    torch.manual_seed(2)
    n = 96
    base = torch.rand(1, 3, n, n).to(DEVICE)
    copy = base + 0.2                      # 定数オフセットで「浮かせた」copy
    mask = torch.zeros(1, 1, n, n).to(DEVICE)

    before = (_block_means(copy, BLOCK) - _block_means(base, BLOCK)).abs().mean()
    out = photometric_align(copy, base, mask, block=BLOCK)
    after = (_block_means(out, BLOCK) - _block_means(base, BLOCK)).abs().mean()

    assert out.shape == copy.shape
    assert float(after) < float(before) / 5.0


# --- テスト3・4: band_composite の帯ごとの出処 ----------------------------------

def _slope_base_copy(n: int = 128):
    """base: 滑らかな傾斜 (x 方向)。copy: その傾斜 + 別の傾斜 (y 方向) + 高周波の縞。"""
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    slope_a = xx / n                                    # base の傾斜
    slope_b = yy / n                                    # copy にだけ足す別の傾斜
    stripes = 0.3 * np.sin(xx / 3.0)                     # 高周波の縞 (周期 ~6px)

    base_np = np.repeat(slope_a[..., None], 3, axis=2)
    copy_np = np.repeat((slope_a + slope_b + stripes)[..., None], 3, axis=2)

    base = torch.from_numpy(base_np.transpose(2, 0, 1)[None]).to(DEVICE)
    copy = torch.from_numpy(copy_np.transpose(2, 0, 1)[None]).to(DEVICE)
    mask = torch.zeros(1, 1, n, n).to(DEVICE)    # 空間重み w=0 (帯選択そのものを見る)
    return base, copy, mask


def test_band_composite_low_freq_follows_base():
    base, copy, mask = _slope_base_copy()
    out = band_composite(base, copy, mask, levels=5, cross=2)

    sigma = 12.0
    out_blur = _gaussian_blur(out, sigma)
    base_blur = _gaussian_blur(base, sigma)
    copy_blur = _gaussian_blur(copy, sigma)

    d_base = (out_blur - base_blur).abs().mean()
    d_copy = (out_blur - copy_blur).abs().mean()
    assert float(d_base) < float(d_copy)


def test_band_composite_high_freq_follows_copy():
    base, copy, mask = _slope_base_copy()
    out = band_composite(base, copy, mask, levels=5, cross=2)

    sigma = 2.0
    hf_out = (out - _gaussian_blur(out, sigma)).var()
    hf_base = (base - _gaussian_blur(base, sigma)).var()
    assert float(hf_out) > float(hf_base)


# --- テスト5・6: 格子の低減とブロック内部のにじみ回避 ----------------------------

def _rich_texture(n: int, seed: int) -> torch.Tensor:
    """相関のあるランダムテクスチャ (`test_feather.py` と同じ構成)。
    純粋な白色ノイズだと不連続の測定がノイズそのもので支配されるため、
    軽くぼかして局所相関を持たせる。"""
    rng = np.random.default_rng(seed)
    arr = rng.normal(0, 1.0, (n, n, 3)).astype(np.float32)
    t = torch.from_numpy(arr.transpose(2, 0, 1)[None]).to(DEVICE)
    return F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, stride=1)


def _checker_paste(n: int, block: int, seed: int):
    """ブロックごとに全く違うオフセットを市松模様に割り当てた `paste` 出力を作る
    (`test_feather.py` の `_seam_energies` と同じ人工的な構成)。"""
    img = _rich_texture(n, seed=seed)
    mask = torch.zeros(1, 1, n, n).to(DEVICE)
    nb = n // block
    offs = [(24, 18), (-24, -18)]
    checker = (torch.arange(nb)[:, None] + torch.arange(nb)[None, :]) % 2
    idx = checker.view(1, 1, nb, nb).long()
    out_paste = paste(img, mask, idx, offs)
    return img, mask, out_paste


def _seam_energy(out: torch.Tensor, block: int) -> float:
    """`block` の倍数位置にある内部境界をまたぐ隣接画素差の二乗和 (x/y 両方向)。"""
    _, _, h, w = out.shape
    total = 0.0
    for x in range(block, w, block):
        total += float(((out[:, :, :, x] - out[:, :, :, x - 1]) ** 2).sum())
    for y in range(block, h, block):
        total += float(((out[:, :, y, :] - out[:, :, y - 1, :]) ** 2).sum())
    return total


def _interior_mask(n: int, block: int, margin: int = 6) -> torch.Tensor:
    """ブロック境界から `margin` 画素以上離れた「ブロック内部」画素の bool マスク。"""
    idx = torch.arange(n)
    local = idx % block
    inside_1d = (local >= margin) & (local <= block - 1 - margin)
    inside = inside_1d[:, None] & inside_1d[None, :]
    return inside


def _seam_and_interior_var(n: int = 128, block: int = BLOCK, seed: int = 11):
    img, mask, out_paste = _checker_paste(n, block, seed)
    out_band = band_composite(img, out_paste, mask, levels=5, cross=2)

    e_paste = _seam_energy(out_paste, block)
    e_band = _seam_energy(out_band, block)

    sigma = 2.0
    hf_paste = out_paste - _gaussian_blur(out_paste, sigma)
    hf_band = out_band - _gaussian_blur(out_band, sigma)
    inside = _interior_mask(n, block).to(DEVICE)
    inside4 = inside[None, None].expand(1, 3, n, n)
    var_paste = float(hf_paste[inside4].var())
    var_band = float(hf_band[inside4].var())

    return e_paste, e_band, var_paste, var_band


def test_band_composite_reduces_seams():
    """継ぎ目 (ブロック境界をまたぐ隣接画素差の二乗和) が `paste` より小さいこと。
    `band_composite` の存在理由そのもの。"""
    e_paste, e_band, _, _ = _seam_and_interior_var()
    assert e_band < e_paste


def test_band_composite_preserves_interior_texture():
    """ブロック内部 (境界から離れた画素) の高周波分散が `paste` の 70% 以上を
    保っていること。`paste_feather` はここで失格した (境界だけでなく内部まで
    ぼかしていた) — `band_composite` は空間重みが 0 の領域では帯を減衰させずに
    そのまま通すので、内部のテクスチャはほぼ保たれるはず。"""
    _, _, var_paste, var_band = _seam_and_interior_var()
    assert var_band >= 0.7 * var_paste


# --- テスト7: 既定経路の不変性 --------------------------------------------------

def test_inpaint_default_matches_band_false_photo_false():
    n = 64
    img = _rich_texture(n, seed=3)
    mask = torch.zeros(1, 1, n, n).to(DEVICE)
    mask[:, :, 20:44, 20:44] = 1.0
    q = F.avg_pool2d(F.pad(img, (4, 4, 4, 4), mode="replicate"), 9, stride=1)

    out_default = inpaint(img, mask, query=q)
    out_explicit = inpaint(img, mask, query=q, band=False, photo=False)
    assert torch.equal(out_default, out_explicit)

    out_default_noquery = inpaint(img, mask)
    out_explicit_noquery = inpaint(img, mask, band=False, photo=False)
    assert torch.equal(out_default_noquery, out_explicit_noquery)


if __name__ == "__main__":
    e_paste, e_band, var_paste, var_band = _seam_and_interior_var()
    print(f"seam energy: paste={e_paste:.6f}  band={e_band:.6f}")
    print(f"interior hf var: paste={var_paste:.6f}  band={var_band:.6f}  "
          f"retention={var_band / var_paste:.3f}")
