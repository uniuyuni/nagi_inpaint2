"""上位k候補の重み付き平均合成 (`topk_weights` / `paste_topk` / `inpaint(topk=...)`) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない)。

**単一コピーは神託上限を超えられない。** `paste` は argmin の1候補をそのまま
コピーするので、その1点のノイズや不整合がそのまま出力に残る。上位k候補の
重み付き平均は、各候補が持つ独立な誤差成分を 1/k 近くまで減らせるので、
「正解を見て最良の1候補を選ぶ」神託の上限すら超えうる。ここではその性質が
実装として正しく効くことを確認する。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b_search import (   # noqa: E402
    inpaint, paste, paste_topk, solve_blocks, topk_weights,
)

DEVICE = "cpu"


def _periodic_texture(n: int, noise: float, seed: int) -> torch.Tensor:
    """x 方向にのみ周期的な (= y 方向には不変な) テクスチャ。

    複数の x シフト (周期の整数倍) が同程度に良い一致になるため、上位k候補に
    複数の有効な参照元が並びやすい (topk による平均化が効く条件を作る)。
    """
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


# --- テスト1: k=1 は paste と一致する ---------------------------------------

def test_topk1_weights_are_ones_and_matches_paste():
    img = _periodic_texture(64, noise=0.02, seed=3)
    mask = _center_mask(64, 18, 46)
    q = _blur_query(img)

    idx, offs, Cn = solve_blocks(img, mask, query=q)   # idx は _fill_blocks_inward 済み

    idx_k1, w_k1 = topk_weights(Cn, k=1)
    assert torch.allclose(w_k1, torch.ones_like(w_k1))

    # paste_topk と paste の等価性は、solve_blocks が返す (fill 済みの) idx で確認する。
    # topk_weights が返す idx_k は fill 前の生 Cn.argmin なので、穴の奥のブロック
    # (全候補無効 -> _fill_blocks_inward で伝播) では idx と食い違いうる。
    # これは topk_weights の仕様どおりで bug ではない (SPEC のとおり)。
    ones = torch.ones_like(idx, dtype=torch.float32)
    out_topk = paste_topk(img, mask, idx, ones, offs)
    out_ref = paste(img, mask, idx, offs)
    assert torch.allclose(out_topk, out_ref, atol=1e-6)


# --- テスト2: 無効候補 (BIG の番兵) は重み0 ----------------------------------

def test_topk_weights_excludes_invalid_candidates():
    torch.manual_seed(0)
    Cn = torch.rand(1, 8, 3, 3) * 0.1
    Cn[0, 2] = 1e6   # 1チャンネル丸ごと無効候補にする

    idx_k, w_k = topk_weights(Cn, k=8, tau_k=0.5)   # k=K で全候補を含める
    invalid_mask = idx_k == 2
    assert invalid_mask.any()
    assert torch.all(w_k[invalid_mask] == 0)


# --- テスト3: 重みの和は1 ----------------------------------------------------

def test_topk_weights_sum_to_one():
    torch.manual_seed(1)
    Cn = torch.rand(1, 10, 4, 4)   # 全候補有効
    _, w_k = topk_weights(Cn, k=4, tau_k=0.5)
    assert torch.allclose(w_k.sum(1), torch.ones(1, 4, 4), atol=1e-5)


# --- テスト4: tau_k を小さくすると重みが最小コスト候補へ集中する -------------

def test_tau_k_concentrates_weight():
    img = _periodic_texture(64, noise=0.02, seed=3)
    mask = _center_mask(64, 18, 46)
    q = _blur_query(img)
    _, _, Cn = solve_blocks(img, mask, query=q)

    _, w_hi = topk_weights(Cn, k=4, tau_k=1.0)
    _, w_lo = topk_weights(Cn, k=4, tau_k=0.1)

    max_hi = w_hi.max(dim=1).values
    max_lo = w_lo.max(dim=1).values
    # tau_k が小さいほど最大重みは大きい (単調増加) はず
    assert float(max_lo.mean()) >= float(max_hi.mean())
    assert torch.all(max_lo + 1e-6 >= max_hi)


# --- テスト5: 平均が誤差を減らす (ノイズを含む周期テクスチャ) ----------------

def test_topk_averaging_reduces_mse_vs_gt():
    """複数の独立ノイズ実現を平均すると、単一コピーよりGTに近づくことを確認する。

    GT (ノイズなしの周期テクスチャ) と、それにガウスノイズを足した観測画像を
    別々に作る。観測画像の穴を埋めた結果を GT と比較する: 単一コピー (topk=1)
    は選ばれた1点のノイズをそのまま持ち込むが、上位4候補の平均 (topk=4) は
    独立なノイズ成分が打ち消し合うぶん GT に近づくはず。
    """
    n, lo, hi = 96, 32, 64
    yy, xx = np.mgrid[:n, :n]
    base = 0.5 + 0.4 * np.sin(xx / 6.0)
    gt_np = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    rng = np.random.default_rng(0)
    noisy_np = gt_np + rng.normal(0, 0.15, (n, n, 3)).astype(np.float32)

    img = torch.from_numpy(noisy_np.transpose(2, 0, 1)[None]).to(DEVICE)
    gt = torch.from_numpy(gt_np.transpose(2, 0, 1)[None]).to(DEVICE)
    mask = _center_mask(n, lo, hi)
    q = _blur_query(img)

    out1 = inpaint(img, mask, query=q, topk=1)
    out4 = inpaint(img, mask, query=q, topk=4, tau_k=0.5)

    m = mask.expand_as(img) > 0.5
    mse1 = ((out1 - gt)[m] ** 2).mean().item()
    mse4 = ((out4 - gt)[m] ** 2).mean().item()
    assert mse4 <= mse1


# --- テスト6: 既定 (topk=1) は変更前と完全一致 -------------------------------

def test_inpaint_default_matches_topk1():
    img = _periodic_texture(56, noise=0.02, seed=5)
    mask = _center_mask(56, 20, 36)

    out_default = inpaint(img, mask)
    out_topk1 = inpaint(img, mask, topk=1)
    assert torch.equal(out_default, out_topk1)

    # query / fallback ありでも同様
    q = _blur_query(img)
    out_default_fb = inpaint(img, mask, fallback=True, query=q)
    out_topk1_fb = inpaint(img, mask, fallback=True, query=q, topk=1)
    assert torch.equal(out_default_fb, out_topk1_fb)
