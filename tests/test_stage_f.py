"""Stage F (全解像度 FFC 補完、実現可能性検証) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない)。

確認すること:
  1. forward が入力と同じ解像度を返す (StageA と違い縮小しないので、これが崩れると
     エンコーダ/デコーダの段数不一致にすぐ気付ける)
  2. パラメータ数が 15M 以下 (M1 で学習可能な範囲に収まっているか)
  3. BatchNorm を含まない (マスク入力で統計が壊れた過去の失敗を再発させない)
  4. buffers が 0 個 (InstanceNorm は running 統計を持たない構成であることの担保。
     `|Δw|max` を named_parameters() のみで計算する前提でもある)
  5. 損失が有限で backward が通り、勾配ノルムが 0 でない (学習が実際に進む土台)
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_f import StageF          # noqa: E402
from train.train_stage_f import grad_loss, step_loss  # noqa: E402

DEVICE = "cpu"


def _sample(n: int, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(batch, 3, n, n, generator=g, device=DEVICE)
    mask = (torch.rand(batch, 1, n, n, generator=g, device=DEVICE) > 0.7).float()
    return img, mask


# --- 1. 解像度を保つこと ---------------------------------------------------

def test_forward_preserves_resolution_256():
    model = StageF().to(DEVICE)
    img, mask = _sample(256)
    with torch.no_grad():
        out = model(img, mask)
    assert out.shape == img.shape


def test_forward_preserves_resolution_384():
    model = StageF().to(DEVICE)
    img, mask = _sample(384)
    with torch.no_grad():
        out = model(img, mask)
    assert out.shape == img.shape


# --- 2. パラメータ数 --------------------------------------------------------

def test_param_count_under_15m():
    model = StageF().to(DEVICE)
    n_par = sum(q.numel() for q in model.parameters())
    assert n_par <= 15_000_000, f"params={n_par:,} が 15M を超えている"
    assert n_par > 0


# --- 3. BatchNorm を含まないこと --------------------------------------------

def test_no_batchnorm():
    model = StageF().to(DEVICE)
    bn = [name for name, m in model.named_modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))]
    assert bn == [], f"BatchNorm を含んでいる: {bn}"


# --- 4. buffers が 0 個 ------------------------------------------------------

def test_no_buffers():
    model = StageF().to(DEVICE)
    bufs = list(model.named_buffers())
    assert bufs == [], f"buffers を持っている (InstanceNorm 以外が混入?): {[n for n, _ in bufs]}"


# --- 5. 損失が有限で勾配が流れること -----------------------------------------

def test_grad_loss_finite_and_symmetric():
    a, b = _sample(64, seed=1)
    a = a[:, :3]  # 3ch にそろえる (grad_loss は img と同 shape を仮定)
    b = b.expand(-1, 3, -1, -1)
    lg = grad_loss(a, b)
    assert torch.isfinite(lg)
    assert float(lg) >= 0.0


def test_step_loss_finite_and_backward_flows():
    model = StageF().to(DEVICE)
    img, mask = _sample(96, batch=2, seed=2)
    loss, l1 = step_loss(model, img, mask, lam_grad=0.5)
    assert loss is not None
    assert torch.isfinite(loss)
    assert torch.isfinite(l1)

    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters()
                    if p.grad is not None)
    assert grad_norm > 0.0, "backward しても勾配が流れていない"


def test_step_loss_none_when_mask_empty():
    """穴が無ければ損失を定義できない (StageA と同じ規約)。"""
    model = StageF().to(DEVICE)
    img = torch.rand(1, 3, 64, 64, device=DEVICE)
    mask = torch.zeros(1, 1, 64, 64, device=DEVICE)
    loss, l1 = step_loss(model, img, mask, lam_grad=0.5)
    assert loss is None
    assert l1 is None
