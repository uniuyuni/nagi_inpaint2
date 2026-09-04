"""Stage G (残差を Rectified Flow で標本として引く) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない、`tests/test_stage_f.py`
と同じ理由)。

確認すること (`nagi_inpaint2/stage_g.py` の docstring も参照):
  1. forward が (N,3,H,W) を返し、パラメータ数が 3M 以下
  2. BatchNorm を含まない (マスク入力で統計が壊れた過去の失敗を再発させない)
  3. `make_cond` が 8ch を返し、チャンネル順が仕様どおり
  4. `rectified_flow_loss` が有限で、backward で非ゼロ勾配が流れる
  5. `sample_residual` がマスク外で 0 を返す (残差は穴の中だけの量、という契約)
  6. 同じ `generator` なら同じ結果、違えば違う結果 (標本を引いていることの担保。
     決定論的回帰にすり替わっていないかを見る)
  7. **速度場が正しく学習できることの最小確認 (この設計の存在理由)**。
     ごく単純な目標 (固定の縞模様) に対して 300 step 学習し、`sample_residual`
     の出力が目標に近づくこと (相関 0.5 以上) を見る。ここが崩れているなら
     モデル構造や損失式のどこかが壊れており、`train/train_stage_g.py` の
     `--overfit` ゲートより前に検出できる。
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_g import (StageG, make_cond,              # noqa: E402
                                   rectified_flow_loss, sample_residual)

DEVICE = "cpu"


def _sample(n: int, batch: int = 1, cond_ch: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(batch, 3, n, n, generator=g, device=DEVICE)
    t = torch.rand(batch, generator=g, device=DEVICE)
    cond = torch.rand(batch, cond_ch, n, n, generator=g, device=DEVICE)
    return z, t, cond


# --- 1. forward の形とパラメータ数 -------------------------------------------

def test_forward_shape():
    model = StageG().to(DEVICE)
    z, t, cond = _sample(64)
    with torch.no_grad():
        v = model(z, t, cond)
    assert v.shape == z.shape


def test_param_count_under_3m():
    model = StageG().to(DEVICE)
    n_par = sum(q.numel() for q in model.parameters())
    assert n_par <= 3_000_000, f"params={n_par:,} が 3M を超えている"
    assert n_par > 0


# --- 2. BatchNorm を含まないこと --------------------------------------------

def test_no_batchnorm():
    model = StageG().to(DEVICE)
    bn = [name for name, m in model.named_modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))]
    assert bn == [], f"BatchNorm を含んでいる: {bn}"


# --- 3. make_cond のチャンネル順 ---------------------------------------------

def test_make_cond_channel_order():
    n, hw = 2, 16
    sf_out = torch.arange(n * 3 * hw * hw, dtype=torch.float32).reshape(n, 3, hw, hw)
    mask = torch.full((n, 1, hw, hw), 7.0)
    dist = torch.full((n, 1, hw, hw), 16.0)          # /32 -> 0.5
    copy_resid = torch.full((n, 3, hw, hw), -3.0)

    cond = make_cond(sf_out, mask, dist, copy_resid)
    assert cond.shape == (n, 8, hw, hw)

    # ch 0-2: sf_out そのまま
    assert torch.equal(cond[:, 0:3], sf_out)
    # ch 3: mask そのまま (0/1 でなくても make_cond 自体は加工しない)
    assert torch.equal(cond[:, 3:4], mask)
    # ch 4: dist/32 を [0,1] にクリップしたもの
    assert torch.allclose(cond[:, 4:5], torch.full((n, 1, hw, hw), 0.5))
    # ch 5-7: copy_resid そのまま
    assert torch.equal(cond[:, 5:8], copy_resid)


def test_make_cond_dist_clipped_to_unit_range():
    n, hw = 1, 8
    sf_out = torch.zeros(n, 3, hw, hw)
    mask = torch.ones(n, 1, hw, hw)
    dist = torch.full((n, 1, hw, hw), 999.0)          # 32 で割ると 1 を大きく超える
    copy_resid = torch.zeros(n, 3, hw, hw)
    cond = make_cond(sf_out, mask, dist, copy_resid)
    assert float(cond[:, 4:5].max()) <= 1.0
    assert float(cond[:, 4:5].min()) >= 0.0


# --- 4. rectified_flow_loss が有限で勾配が流れること -------------------------

def test_rectified_flow_loss_finite_and_backward_flows():
    model = StageG().to(DEVICE)
    n, hw = 2, 32
    r = torch.randn(n, 3, hw, hw, device=DEVICE)
    sf_out = torch.rand(n, 3, hw, hw, device=DEVICE)
    mask = (torch.rand(n, 1, hw, hw, device=DEVICE) > 0.5).float()
    dist = torch.rand(n, 1, hw, hw, device=DEVICE) * 32
    copy_resid = torch.randn(n, 3, hw, hw, device=DEVICE)
    cond = make_cond(sf_out, mask, dist, copy_resid)

    loss, parts = rectified_flow_loss(model, r, cond)
    assert torch.isfinite(loss)
    assert isinstance(parts, dict)

    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters()
                    if p.grad is not None)
    assert grad_norm > 0.0, "backward しても勾配が流れていない"


# --- 5. sample_residual がマスク外で 0 を返すこと -----------------------------

def test_sample_residual_zero_outside_mask():
    model = StageG().to(DEVICE)
    n, hw = 1, 32
    cond = torch.rand(n, 8, hw, hw, device=DEVICE)
    mask = torch.zeros(n, 1, hw, hw, device=DEVICE)
    mask[:, :, 8:24, 8:24] = 1.0                      # 中央だけ穴

    out = sample_residual(model, cond, mask, steps=4,
                          generator=torch.Generator().manual_seed(0))
    assert out.shape == (n, 3, hw, hw)
    outside = (mask.expand_as(out) < 0.5)
    assert bool((out[outside] == 0).all()), "マスク外が 0 になっていない"


# --- 6. generator による決定性 -----------------------------------------------

def test_sample_residual_deterministic_with_same_generator():
    model = StageG().to(DEVICE)
    n, hw = 1, 32
    cond = torch.rand(n, 8, hw, hw, device=DEVICE)
    mask = torch.ones(n, 1, hw, hw, device=DEVICE)

    out1 = sample_residual(model, cond, mask, steps=4,
                           generator=torch.Generator().manual_seed(42))
    out2 = sample_residual(model, cond, mask, steps=4,
                           generator=torch.Generator().manual_seed(42))
    assert torch.equal(out1, out2), "同じ generator (同じ種) なのに結果が異なる"


def test_sample_residual_varies_with_different_generator():
    model = StageG().to(DEVICE)
    n, hw = 1, 32
    cond = torch.rand(n, 8, hw, hw, device=DEVICE)
    mask = torch.ones(n, 1, hw, hw, device=DEVICE)

    out1 = sample_residual(model, cond, mask, steps=4,
                           generator=torch.Generator().manual_seed(1))
    out2 = sample_residual(model, cond, mask, steps=4,
                           generator=torch.Generator().manual_seed(2))
    assert not torch.equal(out1, out2), \
        "異なる generator なのに同じ結果 (標本を引いていない = 決定論的回帰の疑い)"


# --- 7. 速度場が正しく学習できることの最小確認 ---------------------------------

def test_velocity_field_learns_fixed_stripe_target():
    """固定の縞模様を目標にして 300 step 学習し、サンプルが目標に近づくことを見る。

    `nagi_inpaint2/stage_g.py` の docstring にあるとおり、この設計の前提は
    「Rectified Flow の目的関数は単なる回帰 (速度場の L2) なので学習が L1 と
    同程度に安定する」こと。目標が固定 (乱数ではない1点) なら理論上ほぼ厳密に
    再構成できるはずで、これが 300 step 程度で確認できなければモデル構造や
    損失式のどこかが壊れている。
    """
    torch.manual_seed(0)
    hw = 32
    model = StageG(base=8, cond_ch=8).to(DEVICE)

    # 固定の縞模様 (列方向に正弦波、全チャンネル同じ)。乱数ではない1点なので
    # Rectified Flow は決定的に r = f(noise) を学習できるはず。
    xs = torch.arange(hw, dtype=torch.float32)
    stripe = torch.sin(xs * (2 * 3.14159265 / 8.0))
    r = stripe.view(1, 1, 1, hw).expand(1, 3, hw, hw).clone()

    sf_out = torch.zeros(1, 3, hw, hw)
    mask = torch.ones(1, 1, hw, hw)                   # 全面が穴 (減衰させない)
    dist = torch.zeros(1, 1, hw, hw)
    copy_resid = torch.zeros(1, 3, hw, hw)
    cond = make_cond(sf_out, mask, dist, copy_resid)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(300):
        loss, _ = rectified_flow_loss(model, r, cond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.5, \
        f"loss が十分下がっていない: {losses[0]:.4f} -> {losses[-1]:.4f}"

    out = sample_residual(model, cond, mask, steps=8,
                          generator=torch.Generator().manual_seed(0))
    a = out.flatten()
    b = r.flatten()
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    assert corr >= 0.5, f"サンプルと目標の相関が低すぎる: {corr:.3f}"
