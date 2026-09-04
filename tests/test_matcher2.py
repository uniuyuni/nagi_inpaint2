"""照合コスト学習 v2 (`train/train_matcher2.py`) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない、SPEC の制約)。

3つの観点:

  1. `block_costs` (キャッシュされた少数オフセットだけを評価する高速版) が
     `stage_b_search.solve_blocks(..., query=q)` の対応オフセット列と厳密に
     一致すること。式を分岐させて実装した以上、数値一致で担保しないと
     推論と学習が別物を最適化してしまう。
  2. `pairwise_hinge_loss` が「順位が合っている」ときに小さく、
     「完全に逆順」のときに大きいこと。これが損失として機能する最低条件。
  3. FeatureNet を実際に数十 step 学習させて loss が下がること。
     勾配が本当に流れているかの経験的な確認 (`stage_b.py` の
     「ゼロ初期化で勾配が厳密に0だった」教訓への保険)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b import FeatureNet                     # noqa: E402
from nagi_inpaint2.stage_b_search import BLOCK, PATCH, solve_blocks  # noqa: E402
from train.train_matcher2 import (                               # noqa: E402
    HINGE_MARGIN, MIN_COV, MIN_VALID, block_costs, pairwise_hinge_loss,
)

DEVICE = "cpu"


# --- テスト1: block_costs が solve_blocks の query 分岐と一致する -----------

def test_block_costs_matches_solve_blocks_query_branch():
    torch.manual_seed(0)
    n = 64
    img = torch.rand(1, 3, n, n, device=DEVICE)
    mask = torch.zeros(1, 1, n, n, device=DEVICE)
    mask[:, :, 20:44, 20:44] = 1.0
    # query は Stage A の低周波補完の代用。値は何でもよい (式が一致することだけを見る)。
    q = torch.rand(1, 3, n, n, device=DEVICE)
    net = FeatureNet().to(DEVICE)

    _, offs_full, Cn_full = solve_blocks(img, mask, feat=net, query=q)

    # オフセット部分集合で比較する (2400候補全部を見る必要はない)。
    sub_idx = list(range(0, len(offs_full), 37))
    assert len(sub_idx) >= 20
    offs_sub = [offs_full[i] for i in sub_idx]

    out = block_costs(q, mask, offs_sub, net, block=BLOCK, patch=PATCH)
    expected = Cn_full[0][sub_idx]

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-4)


# --- テスト2: pairwise hinge 損失の性質 --------------------------------------

def _toy_block(n: int = 4):
    """N=4候補・2x2ブロックの手作り誤差テンソルを作る。"""
    h, w = 2, 2
    cov = torch.full((h, w), 0.5)
    usable = torch.ones(n, h, w, dtype=torch.bool)
    err = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(n, 1, 1).expand(n, h, w).clone()
    return cov, usable, err


def test_pairwise_hinge_loss_small_for_correct_order():
    cov, usable, err = _toy_block()
    rng = np.random.default_rng(0)
    cost = err.clone()                      # コストの順位 = 誤差の順位 (正しい)
    loss = pairwise_hinge_loss(cost, err, usable, cov, rng)
    assert loss is not None
    assert float(loss) <= HINGE_MARGIN + 1e-6


def test_pairwise_hinge_loss_large_for_reversed_order():
    cov, usable, err = _toy_block()
    rng = np.random.default_rng(0)
    cost_correct = err.clone()
    cost_reversed = 5.0 - err               # 誤差の順位と完全に逆 (同じ値の集合)
    rng2 = np.random.default_rng(0)
    loss_correct = pairwise_hinge_loss(cost_correct, err, usable, cov, rng)
    loss_reversed = pairwise_hinge_loss(cost_reversed, err, usable, cov, rng2)
    assert loss_reversed is not None
    assert float(loss_reversed) > float(loss_correct)


def test_pairwise_hinge_loss_none_when_no_target_blocks():
    """cov が閾値未満、または有効候補が1つ以下ならペアが作れず None を返す。"""
    h, w = 2, 2
    cov_low = torch.full((h, w), 0.1)       # MIN_COV=0.25 未満
    usable = torch.ones(4, h, w, dtype=torch.bool)
    err = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1, 1).expand(4, h, w).clone()
    cost = err.clone()
    rng = np.random.default_rng(0)
    assert pairwise_hinge_loss(cost, err, usable, cov_low, rng) is None

    cov_ok = torch.full((h, w), 0.5)
    usable_one = torch.zeros(4, h, w, dtype=torch.bool)
    usable_one[0] = True                    # 有効候補が1つしかない
    assert pairwise_hinge_loss(cost, err, usable_one, cov_ok, rng) is None


# --- テスト3: overfit ループで loss が低下する -------------------------------

def _fake_tile(seed: int, n_off: int = 8, size: int = 64, block: int = BLOCK):
    """numpy で自作した偽キャッシュ1タイル分 (query, mask, err, cov, vfrac, offs)。

    query は乱数 (中身の写実性は問わない、学習ループの機構だけを見る)。
    err は候補ごとに独立な乱数で与える。これで FeatureNet の初期コスト順位
    (query のランダムなテクスチャ由来) と err の順位はほぼ無相関になり、
    30 step の過学習でその特定2タイルに対する順位を合わせられるかを試せる。
    """
    rng = np.random.default_rng(seed)
    query = torch.from_numpy(rng.uniform(0, 1, (1, 3, size, size)).astype(np.float32))
    mask = torch.zeros(1, 1, size, size)
    mask[:, :, size // 4: size * 3 // 4, size // 4: size * 3 // 4] = 1.0

    offs: list[tuple[int, int]] = []
    while len(offs) < n_off:
        o = (int(rng.integers(-8, 9)) * 4, int(rng.integers(-8, 9)) * 4)
        if o != (0, 0) and o not in offs:
            offs.append(o)

    hb, wb = size // block, size // block
    err = torch.from_numpy(rng.uniform(0.01, 1.0, size=(n_off, hb, wb)).astype(np.float32))
    vfrac = torch.ones(n_off, hb, wb)
    cov = torch.full((hb, wb), 0.9)
    return query, mask, err, cov, vfrac, offs


def test_overfit_two_tiles_reduces_loss():
    torch.manual_seed(0)
    net = FeatureNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    tiles = [_fake_tile(0), _fake_tile(1)]
    rng = np.random.default_rng(0)

    losses = []
    for step in range(30):
        query, mask, err, cov, vfrac, offs = tiles[step % 2]
        cost = block_costs(query, mask, offs, net, block=BLOCK, patch=PATCH)
        usable = (vfrac > MIN_VALID) & (cov > MIN_COV)[None]
        loss = pairwise_hinge_loss(cost, err, usable, cov, rng)
        assert loss is not None, "テスト用の合成データはペアが作れる設計のはず"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))

    assert len(losses) == 30
    early = float(np.mean(losses[:5]))
    late = float(np.mean(losses[-5:]))
    assert late < early, f"loss が下がっていない: 序盤 {early:.5f} -> 終盤 {late:.5f}"
