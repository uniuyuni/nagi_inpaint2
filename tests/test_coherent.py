"""近傍コヒーレントなオフセット選択 (`coherent_idx` / `inpaint(coherent=...)`) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない)。

**これは平均化ではない。** `paste_topk` (topk) が複数候補を重み付き平均してテクスチャを
ぼかすのに対し、`coherent_idx` はあくまで「1ブロック1候補のコピー」を保ったまま、
どの候補を選ぶかを近傍のオフセットと整合する方向に倒す (PatchMatch の伝播ステップに
相当)。ブロックごとに独立に argmin を取ると、隣接ブロックが無関係な場所からコピーして
構造が寸断される — この事前知識 (隣接ブロックの正解オフセットは近いことが多い) を
コストへの軽いペナルティとして注入する。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b_search import coherent_idx, inpaint, solve_blocks  # noqa: E402

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


# --- テスト1: 恒等性 (自分の現在値が5候補中で厳密に最小コストになるよう作る) ---

def test_identity_when_self_is_strictly_cheapest():
    """`lam=0` でも、ペナルティが無いだけで「自分が最小コスト」とは限らない。

    5候補 (自分 + 上下左右の隣接ブロックの現在値) の中で自分の現在の idx が
    *厳密に* 最小コストになるように `Cn` を人工的に作ってから確認する。
    そうしないと、`lam=0` でも近傍のほうが自分のコストで見て安ければ乗り換わりうる
    (これは仕様どおりであってバグではない)。
    """
    h, w, K = 5, 5, 6
    offs = [(0, 0), (4, 0), (0, 4), (8, 0), (0, 8), (4, 4)]
    torch.manual_seed(0)

    # 各ブロックの現在の idx をバラバラに設定する
    idx = torch.randint(0, K, (1, 1, h, w), dtype=torch.long)

    # Cn を大きな値で埋めたあと、各ブロックの「自分の現在の idx」の位置だけ
    # 極端に小さい値を入れる。これで、自分自身の候補が必ず他の候補
    # (= 他ブロックの idx) より安くなる。
    Cn = torch.full((1, K, h, w), 10.0)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    Cn[0, idx[0, 0], yy, xx] = 0.001

    out = coherent_idx(Cn, offs, idx, lam=0.0, tol=8.0, iters=4)
    assert torch.equal(out, idx)


# --- テスト2: 平滑化の効果 (lam を大きくすると孤立ブロックが吸収される) ---

def test_large_lam_absorbs_isolated_outlier():
    """全ブロックで候補Aがわずかに最小だが、1ブロックだけ候補Bがごく僅かに安い状況。

    `lam` を十分大きくすると、そのブロックも候補Aに吸収されオフセット場が一様になる
    ことを確認する。
    """
    h, w, K = 4, 4, 2
    offs = [(0, 0), (40, 40)]   # A=候補0 (皆と揃う), B=候補1 (画素距離が非常に大きい)
    A, B = 0, 1

    Cn = torch.zeros(1, K, h, w)
    Cn[0, A] = 1.0     # 候補Aのコスト
    Cn[0, B] = 1.0
    Cn[0, B, 2, 2] = 0.9   # 1ブロックだけ候補Bがわずかに安い (差 0.1)

    # 初期 idx は全ブロック候補Aを選んでいる状態から出発 (solve_blocks の argmin相当)
    idx0 = torch.zeros(1, 1, h, w, dtype=torch.long)
    idx0[0, 0, 2, 2] = B

    # lam=0 では孤立ブロックはコストで勝る B のまま (乗り換わらない)
    out0 = coherent_idx(Cn, offs, idx0, lam=0.0, tol=8.0, iters=4)
    assert int(out0[0, 0, 2, 2]) == B

    # lam を大きくしていくと、近傍 (すべて A) との距離ペナルティが B の
    # わずかなコスト優位 (0.1) を上回り、B が A に吸収される。
    # s (スケール正規化) はこの設定でほぼ 1.0 なので、解析的には
    # score(B)=0.9+lam*1.0 と score(A)=1.0 が lam=0.1 で釣り合う。
    lam_found = None
    for lam in [0.01, 0.05, 0.1, 0.11, 0.15, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        out = coherent_idx(Cn, offs, idx0, lam=lam, tol=8.0, iters=4)
        if int(out[0, 0, 2, 2]) == A:
            lam_found = lam
            break
    assert lam_found is not None, "どの lam でも孤立ブロックが吸収されなかった"
    # 吸収された場合、オフセット場は完全に一様 (全ブロック A) になっているはず
    out = coherent_idx(Cn, offs, idx0, lam=lam_found, tol=8.0, iters=4)
    assert torch.all(out == A)


# --- テスト3: 無効ブロックの保護 ---

def test_invalid_block_is_not_updated():
    h, w, K = 4, 4, 5
    offs = [(0, 0), (1, 0), (0, 1), (2, 2), (3, 1)]
    torch.manual_seed(2)
    Cn = torch.rand(1, K, h, w)
    # 中央のブロックを全候補無効 (穴の奥) にする
    Cn[0, :, 1, 1] = 1e6
    Cn[0, :, 2, 2] = 1e6

    idx = torch.randint(0, K, (1, 1, h, w), dtype=torch.long)
    out = coherent_idx(Cn, offs, idx, lam=0.5, tol=8.0, iters=4)

    assert int(out[0, 0, 1, 1]) == int(idx[0, 0, 1, 1])
    assert int(out[0, 0, 2, 2]) == int(idx[0, 0, 2, 2])


# --- テスト4: 形状と型 ---

def test_shape_and_dtype():
    img = _periodic_texture(64, noise=0.02, seed=3)
    mask = _center_mask(64, 18, 46)
    q = _blur_query(img)
    idx, offs, Cn = solve_blocks(img, mask, query=q)

    out = coherent_idx(Cn, offs, idx, lam=0.5, tol=8.0, iters=4)
    assert out.shape == idx.shape
    assert out.dtype == torch.long


# --- テスト5: 既定 (coherent=False) は変更前と完全一致 -----------------------

def test_inpaint_default_matches_coherent_false():
    img = _periodic_texture(56, noise=0.02, seed=5)
    mask = _center_mask(56, 20, 36)

    out_default = inpaint(img, mask)
    out_coh_false = inpaint(img, mask, coherent=False)
    assert torch.equal(out_default, out_coh_false)

    # query / fallback / topk ありでも同様
    q = _blur_query(img)
    out_default_fb = inpaint(img, mask, fallback=True, query=q, topk=2)
    out_coh_false_fb = inpaint(img, mask, fallback=True, query=q, topk=2, coherent=False)
    assert torch.equal(out_default_fb, out_coh_false_fb)
