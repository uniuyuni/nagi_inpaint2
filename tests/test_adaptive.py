"""穴の太さに応じた画素単位のコピー/StageF 切り替え (`hole_distance` /
`adaptive_blend` / `inpaint(adaptive=...)`) の回帰テスト。

device は "cpu" 固定 (mps は学習ジョブが使用中のため使わない)。

**背景**: 用途val でマスク種別ごとの PSNR (Stage D 後) を比べると、rect
(太い矩形) だけコピーと StageF の優劣が逆転する:

    種別               コピー(K8)   StageF単体
    wire (細い線)        30.20        27.48
    dust (小さい点)      24.47        22.06
    stroke               22.13        21.34
    rect (太い矩形)      19.29        19.43   <- ここだけ逆転

穴が太いと内部画素は既知画素から遠く、コピー元として正しい実画素が存在しない
(目視で rect 内部に 32px の正方形パッチワークやにじみを確認済み)。
`hole_distance` (最も近い既知画素までの距離) と `adaptive_blend` (その距離で
copy と StageF を画素単位で混ぜる) は、この逆転を画素単位で解消するために
追加した。テスト2 (`hole_distance` が実際に「穴の太さ」を測れていること) が
この関数群の存在理由そのものなので、これが無いテストスイートは要件を満たさない。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b_search import (   # noqa: E402
    adaptive_blend, hole_distance, inpaint,
)

DEVICE = "cpu"


def _periodic_texture(n: int, noise: float, seed: int) -> torch.Tensor:
    """x 方向にのみ周期的な (= y 方向には不変な) テクスチャ (他テストと同じ構成)。"""
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


def _row_band_mask(n: int, lo: int, hi: int) -> torch.Tensor:
    """行 [lo, hi) を全列にわたって穴にする帯状マスク。

    帯が全幅にわたるため、帯内の画素から最も近い既知画素までの距離は
    純粋に「帯の上端/下端までの縦方向の距離」だけで決まり、`hole_distance`
    の値を手計算 (`min(row - lo + 1, hi - row)`) で厳密に検算できる。
    """
    m = np.zeros((n, n), dtype=np.float32)
    m[lo:hi, :] = 1.0
    return torch.from_numpy(m[None, None]).to(DEVICE)


def _blur_query(img: torch.Tensor, k: int = 9) -> torch.Tensor:
    """`img` をぼかして query を作る (Stage A / Stage F の出力の代用)。"""
    r = k // 2
    return F.avg_pool2d(F.pad(img, (r, r, r, r), mode="replicate"), k, stride=1)


# --- テスト1: 既定経路の不変性 (adaptive のデフォルトは無効化と同じ) --------------

def test_inpaint_default_matches_adaptive_false():
    img = _periodic_texture(64, noise=0.02, seed=3)
    mask = _center_mask(64, 18, 46)
    q = _blur_query(img)

    out_default = inpaint(img, mask, query=q)
    out_explicit = inpaint(img, mask, query=q, adaptive=False)
    assert torch.equal(out_default, out_explicit)

    # topk >= 2 の経路でも bit 一致すること
    out_topk_default = inpaint(img, mask, query=q, topk=4)
    out_topk_explicit = inpaint(img, mask, query=q, topk=4, adaptive=False)
    assert torch.equal(out_topk_default, out_topk_explicit)

    # query が無ければ adaptive=True を指定しても無視され、従来の経路を通ること
    out_no_query = inpaint(img, mask)
    out_adaptive_no_query = inpaint(img, mask, adaptive=True)
    assert torch.equal(out_no_query, out_adaptive_no_query)


# --- テスト2: hole_distance の正しさ (この関数の存在理由そのもの) ----------------

def test_hole_distance_center_of_band_is_half_width():
    """帯の中心画素の距離が W/2 に近いこと (細い帯・太い帯の両方)。"""
    n = 128
    results = {}
    for width in (2, 20):
        lo = n // 2 - width // 2
        hi = lo + width
        mask = _row_band_mask(n, lo, hi)
        d = hole_distance(mask)

        center_row = (lo + hi - 1) // 2
        center_col = n // 2
        val = float(d[0, 0, center_row, center_col])
        results[width] = val
        assert abs(val - width / 2.0) <= 1.5, (
            f"width={width}: 中心画素の距離 {val} が W/2={width/2.0} から離れすぎ")

    # 手動実行時の実測値確認用 (pytest -s で見える)
    print(f"hole_distance center: W=2 -> {results[2]}, W=20 -> {results[20]}")


# --- テスト3: マスク外は距離 0 --------------------------------------------------

def test_hole_distance_zero_outside_mask():
    n = 64
    mask = _center_mask(n, 20, 44)
    d = hole_distance(mask)
    outside = mask <= 0.5
    assert torch.all(d[outside] == 0)


# --- テスト4: 細い穴では copy が選ばれる ----------------------------------------

def test_adaptive_blend_thin_hole_matches_copy():
    n = 64
    width = 2
    lo = n // 2 - width // 2
    hi = lo + width
    mask = _row_band_mask(n, lo, hi)

    g = torch.Generator().manual_seed(0)
    copy = torch.rand(1, 3, n, n, generator=g).to(DEVICE)
    g2 = torch.Generator().manual_seed(1)
    base = torch.rand(1, 3, n, n, generator=g2).to(DEVICE)

    out = adaptive_blend(copy, base, mask)
    # W=2 なら帯内のどの画素も既知画素まで距離 1 (d0=8 以内) なので、
    # マスク内外を問わず出力全体が copy と一致するはず。
    assert torch.allclose(out, copy, atol=1e-5)


# --- テスト5: 太い穴の中心では base が選ばれる -----------------------------------

def test_adaptive_blend_thick_hole_center_matches_base():
    n = 128
    width = 60
    lo = n // 2 - width // 2
    hi = lo + width
    mask = _row_band_mask(n, lo, hi)

    g = torch.Generator().manual_seed(2)
    copy = torch.rand(1, 3, n, n, generator=g).to(DEVICE)
    g2 = torch.Generator().manual_seed(3)
    base = torch.rand(1, 3, n, n, generator=g2).to(DEVICE)

    out = adaptive_blend(copy, base, mask)
    center_row = (lo + hi - 1) // 2
    center_col = n // 2
    assert torch.allclose(out[:, :, center_row, center_col],
                          base[:, :, center_row, center_col], atol=1e-3)


# --- テスト6: 境界の連続性 (段差が無いこと) --------------------------------------

def test_adaptive_blend_weight_is_continuous():
    """`copy=0`, `base=1` に固定すると `adaptive_blend` の出力はそのまま重み `w`
    になる。`w` は `hole_distance` (1画素動くごとに高々1しか変化しない) の
    線形関数なので、隣接画素差の最大値は `1/(d1-d0)` の高々2倍に収まるはず。
    """
    n = 128
    width = 60
    lo = n // 2 - width // 2
    hi = lo + width
    mask = _row_band_mask(n, lo, hi)
    d0, d1 = 8.0, 24.0

    copy = torch.zeros(1, 3, n, n).to(DEVICE)
    base = torch.ones(1, 3, n, n).to(DEVICE)
    out = adaptive_blend(copy, base, mask, d0=d0, d1=d1)

    dx = (out[:, :, :, 1:] - out[:, :, :, :-1]).abs().max().item()
    dy = (out[:, :, 1:, :] - out[:, :, :-1, :]).abs().max().item()
    limit = 2.0 / (d1 - d0)
    assert dx <= limit + 1e-6
    assert dy <= limit + 1e-6


if __name__ == "__main__":
    # 手動実行用: テスト2 の実測値 (中心画素の距離) を表示する。
    n = 128
    for width in (2, 20):
        lo = n // 2 - width // 2
        hi = lo + width
        mask = _row_band_mask(n, lo, hi)
        d = hole_distance(mask)
        center_row = (lo + hi - 1) // 2
        center_col = n // 2
        print(f"W={width}: center distance = "
              f"{float(d[0, 0, center_row, center_col]):.3f} (W/2={width/2.0})")
