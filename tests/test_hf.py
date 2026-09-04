"""高周波のみの移植 (`hf_transfer` / `inpaint(hf_only=...)`) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない)。

**なぜ「置き換え」から「高周波のみの移植」に変えるか**: `paste`/`paste_topk` は
コピー元の実画素で丸ごと置き換えるため、`query` (StageF の出力) が稼いだ
精度の高い低中周波まで捨ててしまう。実測 (用途val, +StageD) で、置き換え
(k=1) は PSNR が StageF 単体より悪化する一方 (23.49 dB vs 24.66 dB)、
grain/psd (テクスチャの質) は StageF 単体よりずっと良い (0.917/0.166)。
低中周波は `query` 由来、高周波は コピー元の実画素由来という分業にすれば
両方の精度を同時に活かせる、というのがこの変更の狙い。ここではその狙いが
実装として正しく効くこと (低周波は base 由来、高周波は copy 由来) を確認する。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b_search import hf_transfer, inpaint  # noqa: E402

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
    """`img` をぼかして query を作る (Stage A / Stage F の出力の代用)。"""
    r = k // 2
    return F.avg_pool2d(F.pad(img, (r, r, r, r), mode="replicate"), k, stride=1)


def _gaussian_blur_ref(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """テスト用の独立実装。`stage_b_search._gaussian_blur` の内部実装には
    依存せず、「output をガウシアンでぼかすと base に近づく」という数学的な
    性質そのものを検証するために別途書く。
    """
    radius = int(np.ceil(3.0 * sigma))
    xs = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
    out = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    out = F.conv2d(out, kx, groups=c)
    out = F.pad(out, (0, 0, radius, radius), mode="replicate")
    out = F.conv2d(out, ky, groups=c)
    return out


def _const_base_and_hf_copy(n: int = 64, period: float = 3.0, amp: float = 0.2):
    """`base` は定数画像、`copy` は「定数 + 高周波の縞」。

    縞の周期を 3px (ナイキストに近い高周波) にすることで、sigma=2.0 の
    ガウシアン低周波抽出でほぼ完全に「高周波」側へ分類される
    (低周波成分としてほとんど漏れない) 状況を作る。テスト4のガウシアンぼかし
    の実測誤差が小さく安定するように選んだパラメータ。
    """
    yy, xx = np.mgrid[:n, :n]
    base_np = np.full((n, n, 3), 0.5, dtype=np.float32)
    stripe = amp * np.sin(2 * np.pi * xx / period)
    copy_np = base_np + np.repeat(stripe[..., None], 3, axis=2).astype(np.float32)
    base = torch.from_numpy(base_np.transpose(2, 0, 1)[None]).to(DEVICE)
    copy = torch.from_numpy(copy_np.transpose(2, 0, 1)[None]).to(DEVICE)
    mask = _center_mask(n, 20, 44)
    return base, copy, mask


# --- テスト1: 既定経路の不変性 (hf_only 未指定 == hf_only=False) ----------------

def test_hf_only_false_matches_default():
    img = _periodic_texture(64, noise=0.02, seed=3)
    mask = _center_mask(64, 18, 46)
    q = _blur_query(img)

    out_default = inpaint(img, mask, query=q)
    out_explicit = inpaint(img, mask, query=q, hf_only=False)
    assert torch.equal(out_default, out_explicit)


# --- テスト2: query が無ければ hf_only は無視される ----------------------------

def test_hf_only_ignored_without_query():
    img = _periodic_texture(56, noise=0.02, seed=5)
    mask = _center_mask(56, 20, 36)

    out_no_query = inpaint(img, mask)
    out_hf_only = inpaint(img, mask, hf_only=True)
    assert torch.equal(out_no_query, out_hf_only)


# --- テスト3: alpha=0 のとき base と一致する -----------------------------------

def test_hf_transfer_alpha_zero_matches_base():
    base, copy, mask = _const_base_and_hf_copy()
    out = hf_transfer(base, copy, mask, alpha=0.0)
    assert torch.allclose(out, base, atol=1e-6)


# --- テスト4: 低周波は base 由来である -----------------------------------------

def test_hf_transfer_low_freq_comes_from_base():
    base, copy, mask = _const_base_and_hf_copy()
    sigma = 2.0
    out = hf_transfer(base, copy, mask, sigma=sigma, alpha=1.0)

    lo_out = _gaussian_blur_ref(out, sigma)
    assert torch.allclose(lo_out, base, atol=0.02)


# --- テスト5: 高周波は copy 由来である ------------------------------------------

def test_hf_transfer_high_freq_comes_from_copy():
    base, copy, mask = _const_base_and_hf_copy()
    sigma = 2.0
    out = hf_transfer(base, copy, mask, sigma=sigma, alpha=1.0)

    hf_out = out - _gaussian_blur_ref(out, sigma)
    hf_base = base - _gaussian_blur_ref(base, sigma)
    # base は定数なので hf_base の分散は 0 のはず。out には copy の高周波
    # (縞) が乗っているので、その分散は有意に (定数倍以上) 大きい。
    assert hf_out.var().item() > hf_base.var().item() + 1e-4


# --- テスト6: マスク外にも合成が伸びている (paste と同じ契約) -------------------

def test_hf_transfer_extends_outside_mask():
    base, copy, mask = _const_base_and_hf_copy()
    out = hf_transfer(base, copy, mask)

    outside = mask.expand_as(base) <= 0.5
    assert outside.any()
    assert not torch.equal(out[outside], base[outside])
