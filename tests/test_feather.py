"""ブロック貼り付けの重なり + 羽化 (`paste_feather` / `inpaint(feather=...)`) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない)。

**背景**: `paste` は `BLOCK` (32px) 格子を隙間なく敷き詰めて貼るだけなので、隣接
ブロックが無関係な場所からコピーすると格子の境目がそのまま画素の不連続として残る
(目視検証で確認済み。幅のあるマスクの内部で顕著)。`paste_feather` はブロックを
半分ずつ重ねて貼り、ハン窓で重み付き平均することでこの継ぎ目を滑らかにする。

このテストの主眼はテスト4 (継ぎ目が実際に減ること) — `paste_feather` の存在理由
そのものなので、これが無いテストスイートは要件を満たさない。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b_search import (   # noqa: E402
    BLOCK, inpaint, paste, paste_feather, solve_blocks,
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


def _blur_query(img: torch.Tensor, k: int = 9) -> torch.Tensor:
    """`img` をぼかして query を作る (Stage A の低周波補完の代用)。"""
    r = k // 2
    return F.avg_pool2d(F.pad(img, (r, r, r, r), mode="replicate"), k, stride=1)


def _rich_texture(n: int, seed: int) -> torch.Tensor:
    """相関のあるランダムテクスチャ。異なるオフセットで明確に別の見た目になる。

    純粋な白色ノイズだと境界差がノイズそのもので支配され、羽化の効果 (低周波な
    継ぎ目をならす効果) が測りにくい。軽くぼかして局所的な相関を持たせることで、
    「ブロックごとに違う場所からコピーした」ことによる不連続を測りやすくする。
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.0, (n, n, 3)).astype(np.float32)
    t = torch.from_numpy(base.transpose(2, 0, 1)[None]).to(DEVICE)
    return F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, stride=1)


def _seam_energy(out: torch.Tensor, block: int) -> float:
    """`block` の倍数位置にある内部境界をまたぐ隣接画素差の二乗和 (x/y 両方向)。

    ブロック境界の不連続そのものを定量化する指標。値が小さいほど継ぎ目が目立たない。
    """
    _, _, h, w = out.shape
    total = 0.0
    for x in range(block, w, block):
        total += float(((out[:, :, :, x] - out[:, :, :, x - 1]) ** 2).sum())
    for y in range(block, h, block):
        total += float(((out[:, :, y, :] - out[:, :, y - 1, :]) ** 2).sum())
    return total


# --- テスト1: 既定経路の不変性 (feather のデフォルトは無効化と同じ) --------------

def test_inpaint_default_matches_feather_zero():
    img = _periodic_texture(64, noise=0.02, seed=3)
    mask = _center_mask(64, 18, 46)

    out_default = inpaint(img, mask)
    out_f0 = inpaint(img, mask, feather=0.0)
    assert torch.equal(out_default, out_f0)

    # query / fallback / hf_only ありでも bit 一致 (`inpaint` の分岐構造を
    # 書き換えたので、feather<=0 のときにこれらの経路が壊れていないことも確認する)。
    q = _blur_query(img)
    assert torch.equal(inpaint(img, mask, query=q), inpaint(img, mask, query=q, feather=0.0))
    assert torch.equal(inpaint(img, mask, fallback=True, query=q),
                       inpaint(img, mask, fallback=True, query=q, feather=0.0))
    assert torch.equal(inpaint(img, mask, hf_only=True, query=q),
                       inpaint(img, mask, hf_only=True, query=q, feather=0.0))

    # topk >= 2 と feather の併用は feather を無視して従来の topk 経路を通ること。
    out_topk = inpaint(img, mask, query=q, topk=4)
    out_topk_feather = inpaint(img, mask, query=q, topk=4, feather=0.5)
    assert torch.equal(out_topk, out_topk_feather)


# --- テスト2: 形状 ------------------------------------------------------------

def test_paste_feather_shape_matches_img():
    img = _periodic_texture(64, noise=0.02, seed=1)
    mask = _center_mask(64, 20, 44)
    q = _blur_query(img)
    idx, offs, _ = solve_blocks(img, mask, query=q)

    out = paste_feather(img, mask, idx, offs)
    assert out.shape == img.shape


# --- テスト3: overlap=0 は paste と厳密に一致する -----------------------------

def test_paste_feather_overlap_zero_matches_paste():
    img = _periodic_texture(96, noise=0.05, seed=7)
    mask = _center_mask(96, 24, 72)
    q = _blur_query(img)
    idx, offs, _ = solve_blocks(img, mask, query=q)

    out_paste = paste(img, mask, idx, offs)
    out_feather0 = paste_feather(img, mask, idx, offs, overlap=0.0)
    assert torch.allclose(out_feather0, out_paste, atol=1e-5)


# --- テスト4: 継ぎ目が減ること (paste_feather の存在理由そのもの) ----------------

def _seam_energies() -> tuple[float, float]:
    """ブロックごとに全く違うオフセットが選ばれた `idx` を人工的に作り、
    `paste` と `paste_feather` それぞれのブロック境界の継ぎ目エネルギーを返す。
    """
    n = 128
    block = BLOCK
    img = _rich_texture(n, seed=11)
    mask = torch.zeros(1, 1, n, n).to(DEVICE)   # paste/paste_feather は mask を使わない

    nb = n // block   # ブロック数 (一辺)
    # 互いに大きく離れた2つのオフセットを市松模様に割り当てる。
    # -> 隣接ブロックは常にどちらかの他方を参照し、"全く違うオフセット" が
    #    隣り合う条件を満たす。
    offs = [(24, 18), (-24, -18)]
    checker = (torch.arange(nb)[:, None] + torch.arange(nb)[None, :]) % 2
    idx = checker.view(1, 1, nb, nb).long()

    out_paste = paste(img, mask, idx, offs)
    out_feather = paste_feather(img, mask, idx, offs, block=block, overlap=0.5)

    return _seam_energy(out_paste, block), _seam_energy(out_feather, block)


def test_paste_feather_reduces_seams():
    """ブロック境界をまたぐ隣接画素差の二乗和が `paste` より `paste_feather` の
    ほうが小さいことを確認する (x 方向・y 方向の両方をまとめて測る)。
    """
    e_paste, e_feather = _seam_energies()
    assert e_feather < e_paste


# --- テスト5: マスク外にも合成が伸びている (paste と同じ契約) -------------------

def test_paste_feather_extends_outside_mask():
    img = _periodic_texture(96, noise=0.05, seed=9)
    mask = _center_mask(96, 40, 56)   # 小さめの穴 -> マスク外の面積が大きい
    q = _blur_query(img)
    idx, offs, _ = solve_blocks(img, mask, query=q)

    out = paste_feather(img, mask, idx, offs)

    outside = mask.expand_as(img) <= 0.5
    assert outside.any()
    assert not torch.equal(out[outside], img[outside])


if __name__ == "__main__":
    # 手動実行用: テスト4 の実測値 (境界差の二乗和) を表示する。
    e_paste, e_feather = _seam_energies()
    print(f"paste seam energy    = {e_paste:.6f}")
    print(f"feather seam energy  = {e_feather:.6f}")
