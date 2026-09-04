"""StageFW (勝者総取り学習) の回帰テスト。device は "cpu" 固定。

**背景**: `StageF` は L1 で学習しており平均値回帰でぼける (grain 比 0.109、
LaMa は 0.943)。GAN での修正も2回失敗した (勝率 34/60 → 10/60 まで崩壊)。
WTA は K 個の仮説のうち最良の1つにだけ勾配を流すことで「平均を作る」動機を
消す。このファイルはその WTA の仕組みそのもの (対称性が破れているか、
損失が min に近づくか、勾配が勝者に集中するか、死んだヘッドが出ないか) を
確認する。

確認すること:
  1. `StageFW(k=4).forward` の出力が (N,4,3,H,W) で、入力と同じ解像度
  2. `load_backbone` が `StageF` の state_dict を受け取って動作し、
     出力層以外の重みが一致する
  3. 仮説が同一でないこと (初期化直後、および load_backbone 直後)
  4. WTA 損失の性質: 1つだけ正解に近いとき、損失がその1つの損失に近い値に
     なる (単純平均より小さい)
  5. 勾配が勝者に集中すること (勝者ヘッドの出力層勾配ノルムが敗者の5倍以上)
  6. 全ヘッドに非ゼロの勾配が流れること (eps による死の防止)
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_f import StageF                                   # noqa: E402
from nagi_inpaint2.stage_fw import StageFW                                 # noqa: E402
from train.train_stage_fw import (per_hypothesis_loss, relaxed_wta,        # noqa: E402
                                  step_loss_wta)

DEVICE = "cpu"


def _sample(n: int, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(batch, 3, n, n, generator=g, device=DEVICE)
    mask = (torch.rand(batch, 1, n, n, generator=g, device=DEVICE) > 0.5).float()
    return img, mask


# --- 1. forward の形 ---------------------------------------------------------

def test_forward_shape_and_resolution():
    model = StageFW(base=16, k=4).to(DEVICE)
    img, mask = _sample(64, batch=2, seed=0)
    with torch.no_grad():
        out = model(img, mask)
    assert out.shape == (2, 4, 3, 64, 64)


def test_forward_shape_different_k():
    model = StageFW(base=16, k=3).to(DEVICE)
    img, mask = _sample(48, batch=1, seed=1)
    with torch.no_grad():
        out = model(img, mask)
    assert out.shape == (1, 3, 3, 48, 48)


# --- 2. load_backbone ---------------------------------------------------------

def test_load_backbone_matches_except_out_layer():
    base_model = StageF(base=16).to(DEVICE)
    sd = base_model.state_dict()
    model = StageFW(base=16, k=3).to(DEVICE)
    model.load_backbone(sd)
    backbone_params = dict(model.backbone.named_parameters())
    for name, param in base_model.named_parameters():
        if name.startswith("out."):
            continue
        assert torch.equal(param, backbone_params[name]), f"{name} が一致しない"


def test_load_backbone_out_layer_is_tiled_copy_plus_noise():
    """出力層は K 個とも元の重みのコピー + 小さな乱数のはず。"""
    base_model = StageF(base=16).to(DEVICE)
    sd = base_model.state_dict()
    k = 4
    model = StageFW(base=16, k=k).to(DEVICE)
    model.load_backbone(sd)
    ow = sd["out.weight"]                       # (3, 16, 3, 3)
    for i in range(k):
        head = model.backbone.out.weight[i * 3:(i + 1) * 3]
        assert torch.allclose(head, ow, atol=1e-2), "ヘッドが元の重みから離れすぎている"
        assert not torch.equal(head, ow), "乱数が加わっていない (完全一致している)"


# --- 3. 仮説が同一でないこと (対称性が破れている) -------------------------------

def _max_pairwise_diff(out: torch.Tensor, k: int) -> float:
    diffs = [float((out[:, i] - out[:, j]).abs().max())
             for i in range(k) for j in range(i + 1, k)]
    return min(diffs)


def test_hypotheses_are_not_identical_after_init():
    model = StageFW(base=16, k=4).to(DEVICE)
    img, mask = _sample(64, batch=1, seed=3)
    with torch.no_grad():
        out = model(img, mask)
    assert _max_pairwise_diff(out, 4) > 0.0, "K 個の仮説が同一 (対称性が破れていない)"


def test_hypotheses_are_not_identical_after_load_backbone():
    """load_backbone 直後 (ごく小さな乱数を加えた直後) でも仮説が同一でないこと。

    ここで差が0になると、全ヘッドが同じ答えを出し続け WTA が機能しない
    (常に「たまたま先に並んだヘッド」が勝ち続け、実質 L1 と同じになる)。
    """
    base_model = StageF(base=16).to(DEVICE)
    model = StageFW(base=16, k=4).to(DEVICE)
    model.load_backbone(base_model.state_dict())
    img, mask = _sample(64, batch=1, seed=4)
    with torch.no_grad():
        out = model(img, mask)
    assert _max_pairwise_diff(out, 4) > 0.0


# --- 4. WTA 損失の性質: min に近く、単純平均より小さい ---------------------------

def test_relaxed_wta_close_to_min_not_mean():
    per_k = torch.tensor([[0.01, 1.0, 1.0, 1.0]])   # 1つだけ正解に近い
    eps = 0.05
    loss, winner, _w = relaxed_wta(per_k, eps=eps)
    assert int(winner[0]) == 0
    mean_loss = float(per_k.mean())
    min_loss = float(per_k.min())
    assert float(loss[0]) < mean_loss, "単純平均より小さくなっていない"
    # eps が小さいので min にかなり近いはず
    assert abs(float(loss[0]) - min_loss) < 0.2, \
        f"min ({min_loss}) から離れすぎている: {float(loss[0])}"


def test_per_hypothesis_loss_matches_step_loss_shape():
    model = StageFW(base=16, k=4).to(DEVICE)
    img, mask = _sample(48, batch=2, seed=7)
    with torch.no_grad():
        pred = model(img, mask)
    per_k = per_hypothesis_loss(pred, img, mask, lam_grad=0.5)
    assert per_k is not None
    assert per_k.shape == (2, 4)
    assert torch.isfinite(per_k).all()


def test_per_hypothesis_loss_none_when_mask_empty():
    model = StageFW(base=16, k=4).to(DEVICE)
    img = torch.rand(1, 3, 48, 48, device=DEVICE)
    mask = torch.zeros(1, 1, 48, 48, device=DEVICE)
    with torch.no_grad():
        pred = model(img, mask)
    assert per_hypothesis_loss(pred, img, mask, lam_grad=0.5) is None


# --- 5. 勾配が勝者に集中すること ------------------------------------------------

def test_gradient_concentrates_on_winner_head():
    torch.manual_seed(0)
    model = StageFW(base=16, k=4).to(DEVICE)
    img, mask = _sample(48, batch=1, seed=5)
    loss, info = step_loss_wta(model, img, mask, lam_grad=0.5, eps=0.05)
    assert loss is not None
    model.zero_grad(set_to_none=True)
    loss.backward()

    winner = int(info["winner"][0])
    grad = model.backbone.out.weight.grad
    assert grad is not None
    k = model.k
    per_head = grad.view(k, 3, *grad.shape[1:]).abs().sum(dim=(1, 2, 3, 4))
    winner_norm = float(per_head[winner])
    loser_norms = [float(per_head[i]) for i in range(k) if i != winner]
    assert all(winner_norm > 5 * ln for ln in loser_norms), \
        f"勝者 {winner_norm} vs 敗者 {loser_norms} — 勾配が勝者に集中していない"


# --- 6. 全ヘッドに非ゼロ勾配が流れる (eps による死の防止) --------------------------

def test_all_heads_get_nonzero_gradient():
    torch.manual_seed(1)
    model = StageFW(base=16, k=4).to(DEVICE)
    img, mask = _sample(48, batch=4, seed=6)   # バッチを増やして全ヘッドが勝つ機会を作る
    loss, info = step_loss_wta(model, img, mask, lam_grad=0.5, eps=0.05)
    assert loss is not None
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad = model.backbone.out.weight.grad
    assert grad is not None
    k = model.k
    per_head = grad.view(k, 3, *grad.shape[1:]).abs().sum(dim=(1, 2, 3, 4))
    assert all(float(n) > 0 for n in per_head), \
        f"勾配が来ていないヘッドがある (死んでいる): {per_head.tolist()}"
