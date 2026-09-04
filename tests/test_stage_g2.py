"""Stage G2 (残差の局所スケール正規化) の回帰テスト。

device は "cpu" 固定 (`tests/test_stage_g.py` と同じ理由: mps は他ジョブが
使用中のため使わない)。

**効く理屈と効かない条件は `nagi_inpaint2/stage_g.py` の `residual_scale` 周辺の
docstring を参照** (このタスクの設計指示から直接引用してある)。要約:

  - `StageG` (Rectified Flow で残差を標本として引く) は 3,000 step 学習した
    結果、テクスチャは生成できた (grain 0.63 -> 1.27) が、**平坦な領域に斑模様
    を作った** (黄色いボケの例では、薄い楕円だったものが濃いオリーブ色の塊+
    黒点になった)。原因は、平坦な領域では正解の残差がほぼ 0 なのに、フローは
    全振幅のノイズから積分を始めるため。
  - 処方は、残差を局所的な期待振幅 `s` (= `residual_scale`) で正規化して学習
    すること。平坦な領域では `s ≈ 0` なので、サンプリング後に `s` を掛け戻せば
    出力も自動的に 0 になる。
  - **効かない条件 (漏洩)**: `s` の推定に穴の中の正解を使ってはならない。
    使ってしまうと学習時だけ性能が出て推論で崩れる。このファイルのテスト3が
    それを直接検出する。

確認すること:
  1. `residual_scale` の出力が (N,1,H,W)、全画素で有限かつ非負。
  2. 平坦な画像では `s` が小さく、テクスチャ画像では大きい (比 5倍以上) —
     この関数の存在理由そのものなので必ず確認する。
  3. **漏洩がないことの担保 (最重要)**: 同じ画像で穴の中身だけを変えた2枚
     (穴の外は完全に同一) で `residual_scale` の出力が完全一致すること。
  4. `make_cond9` が 9ch を返し、チャンネル順が仕様どおり。
  5. `sample_residual_scaled` が、`scale` を 0 にした領域で 0 を返すこと
     (平坦部で出力が消えることの担保)。
  6. `rectified_flow_loss_scaled` が有限で backward できること。
  7. 既存の `rectified_flow_loss` / `sample_residual` / `make_cond` の挙動が
     G2 の追加によって変わっていないこと (回帰確認)。
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_g import (StageG, make_cond, make_cond9,        # noqa: E402
                                   rectified_flow_loss, rectified_flow_loss_scaled,
                                   residual_scale, sample_residual,
                                   sample_residual_scaled)

DEVICE = "cpu"


def _checker(n: int, batch: int = 1, freq: int = 4, amp: float = 1.0) -> torch.Tensor:
    """(batch,3,n,n) の市松模様 (高周波テクスチャの代用)。"""
    ys = torch.arange(n).view(n, 1)
    xs = torch.arange(n).view(1, n)
    checker = (((ys // freq) + (xs // freq)) % 2).float() * 2.0 - 1.0   # +-1
    img = checker[None, None].expand(batch, 3, n, n).clone() * amp
    return img


def _flat(n: int, batch: int = 1, value: float = 0.3) -> torch.Tensor:
    """(batch,3,n,n) の平坦画像 (定数)。"""
    return torch.full((batch, 3, n, n), value)


def _hole_mask(n: int, batch: int = 1, size: int | None = None) -> torch.Tensor:
    """中央に矩形の穴を持つ (batch,1,n,n) マスク。"""
    if size is None:
        size = n // 2
    mask = torch.zeros(batch, 1, n, n)
    y0 = (n - size) // 2
    x0 = (n - size) // 2
    mask[:, :, y0:y0 + size, x0:x0 + size] = 1.0
    return mask


# --- 1. residual_scale の形と有限性・非負性 -----------------------------------

def test_residual_scale_shape_finite_nonneg():
    n = 64
    img = _checker(n)
    mask = _hole_mask(n)
    s = residual_scale(img, mask)
    assert s.shape == (1, 1, n, n)
    assert torch.isfinite(s).all(), "全画素で有限であること"
    assert bool((s >= 0).all()), "振幅なので非負であること"


def test_residual_scale_finite_even_when_fully_masked():
    """穴が全面を覆う (既知画素が皆無) 病的なケースでも有限値を返すこと。"""
    n = 32
    img = _checker(n)
    mask = torch.ones(1, 1, n, n)
    s = residual_scale(img, mask)
    assert torch.isfinite(s).all()


# --- 2. 平坦画像とテクスチャ画像で s が大きく異なること (存在理由そのもの) ------

def test_residual_scale_texture_vs_flat_ratio():
    n = 64
    mask = _hole_mask(n)

    s_flat = residual_scale(_flat(n), mask)
    s_tex = residual_scale(_checker(n, amp=1.0), mask)

    flat_amp = float(s_flat.mean())
    tex_amp = float(s_tex.mean())
    assert flat_amp < 1e-6, f"平坦画像の s がほぼ0であるべき: {flat_amp}"
    assert tex_amp > 0.0
    ratio = tex_amp / max(flat_amp, 1e-8)
    assert ratio >= 5.0, (
        f"テクスチャ画像/平坦画像の s の比が5倍未満: {ratio:.2f} "
        f"(flat={flat_amp:.6f}, tex={tex_amp:.6f})")


# --- 3. 漏洩がないことの担保 (最重要) -----------------------------------------

def test_residual_scale_no_leakage_from_hole_interior():
    """穴の中身だけを変えた2枚 (穴の外は完全に同一) で出力が完全一致すること。

    一致しなければ穴の中の情報が推定に漏れている ("効かない条件" そのもの)。
    """
    n = 96
    torch.manual_seed(0)
    base = torch.randn(1, 3, n, n)          # 穴の外側の "地の画像" (テクスチャあり)
    mask = _hole_mask(n, size=40)

    img_a = base.clone()
    img_b = base.clone()

    hole = mask > 0.5
    hole3 = hole.expand_as(base)
    # 穴の中身だけを、まったく違う分布 (定数 / 別乱数 / 極端な値) に差し替える
    img_a = torch.where(hole3, torch.full_like(img_a, 5.0), img_a)
    img_b = torch.where(hole3, torch.randn_like(img_b) * 50.0 - 3.0, img_b)

    # 穴の外は完全に同一であることを前提として確認しておく
    outside = ~hole3
    assert torch.equal(img_a[outside], img_b[outside])
    assert not torch.equal(img_a[hole3], img_b[hole3])

    s_a = residual_scale(img_a, mask)
    s_b = residual_scale(img_b, mask)
    assert torch.equal(s_a, s_b), (
        "residual_scale の出力が穴の中身に依存している (漏洩) — "
        f"max diff = {float((s_a - s_b).abs().max())}")


def test_residual_scale_no_leakage_multiple_trials():
    """複数のマスク形状・シードで漏洩なしを確認する (単発の偶然一致を排除)。"""
    n = 80
    for seed in range(4):
        torch.manual_seed(seed)
        base = torch.randn(1, 3, n, n)
        size = 16 + seed * 12
        mask = _hole_mask(n, size=size)
        hole3 = (mask > 0.5).expand_as(base)

        img_a = torch.where(hole3, torch.zeros_like(base), base)
        img_b = torch.where(hole3, torch.rand_like(base) * 10.0, base)

        s_a = residual_scale(img_a, mask)
        s_b = residual_scale(img_b, mask)
        assert torch.equal(s_a, s_b), f"seed={seed} で漏洩を検出"


# --- 4. make_cond9 のチャンネル順 ---------------------------------------------

def test_make_cond9_channel_order():
    n, hw = 2, 16
    sf_out = torch.arange(n * 3 * hw * hw, dtype=torch.float32).reshape(n, 3, hw, hw)
    mask = torch.full((n, 1, hw, hw), 7.0)
    dist = torch.full((n, 1, hw, hw), 16.0)          # /32 -> 0.5
    copy_resid = torch.full((n, 3, hw, hw), -3.0)
    scale = torch.full((n, 1, hw, hw), 2.5)

    cond9 = make_cond9(sf_out, mask, dist, copy_resid, scale)
    assert cond9.shape == (n, 9, hw, hw)

    cond8 = make_cond(sf_out, mask, dist, copy_resid)
    assert torch.equal(cond9[:, :8], cond8), "ch0-7 は既存 make_cond と一致するべき"
    assert torch.equal(cond9[:, 8:9], scale), "ch8 は scale そのもの"


# --- 5. sample_residual_scaled が scale==0 の領域で 0 を返すこと ---------------

def test_sample_residual_scaled_zero_where_scale_zero():
    model = StageG(cond_ch=9).to(DEVICE)
    n, hw = 1, 32
    cond = torch.rand(n, 9, hw, hw, device=DEVICE)
    mask = torch.ones(n, 1, hw, hw, device=DEVICE)     # 全面が穴
    scale = torch.zeros(n, 1, hw, hw, device=DEVICE)   # 平坦部を模して scale=0

    out = sample_residual_scaled(model, cond, mask, scale, steps=4,
                                 generator=torch.Generator().manual_seed(0))
    assert out.shape == (n, 3, hw, hw)
    assert torch.isfinite(out).all()
    assert bool((out == 0).all()), "scale=0 の領域で出力が0になっていない"


def test_sample_residual_scaled_nonzero_where_scale_nonzero():
    """対照実験: scale が非ゼロなら (一般には) 出力も非ゼロになりうること。"""
    model = StageG(cond_ch=9).to(DEVICE)
    n, hw = 1, 32
    cond = torch.rand(n, 9, hw, hw, device=DEVICE)
    mask = torch.ones(n, 1, hw, hw, device=DEVICE)
    scale = torch.full((n, 1, hw, hw), 3.0, device=DEVICE)

    out = sample_residual_scaled(model, cond, mask, scale, steps=4,
                                 generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out).all()
    # 初期化直後のランダムな重みでも、scale=3 を掛ければほぼ確実にどこかが非ゼロ
    assert bool((out != 0).any())


# --- 6. rectified_flow_loss_scaled が有限で backward できること ----------------

def test_rectified_flow_loss_scaled_finite_and_backward_flows():
    model = StageG(cond_ch=9).to(DEVICE)
    n, hw = 2, 32
    r = torch.randn(n, 3, hw, hw, device=DEVICE)
    sf_out = torch.rand(n, 3, hw, hw, device=DEVICE)
    mask = (torch.rand(n, 1, hw, hw, device=DEVICE) > 0.5).float()
    dist = torch.rand(n, 1, hw, hw, device=DEVICE) * 32
    copy_resid = torch.randn(n, 3, hw, hw, device=DEVICE)
    scale = torch.rand(n, 1, hw, hw, device=DEVICE) * 0.5 + 0.01
    cond = make_cond9(sf_out, mask, dist, copy_resid, scale)

    loss, parts = rectified_flow_loss_scaled(model, r, cond, scale)
    assert torch.isfinite(loss)
    assert isinstance(parts, dict)

    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters()
                    if p.grad is not None)
    assert grad_norm > 0.0, "backward しても勾配が流れていない"


def test_rectified_flow_loss_scaled_handles_all_zero_scale():
    """scale が全域0 (病的ケース) でもゼロ除算で発散しないこと。"""
    model = StageG(cond_ch=9).to(DEVICE)
    n, hw = 1, 16
    r = torch.randn(n, 3, hw, hw, device=DEVICE)
    sf_out = torch.rand(n, 3, hw, hw, device=DEVICE)
    mask = torch.ones(n, 1, hw, hw, device=DEVICE)
    dist = torch.zeros(n, 1, hw, hw, device=DEVICE)
    copy_resid = torch.zeros(n, 3, hw, hw, device=DEVICE)
    scale = torch.zeros(n, 1, hw, hw, device=DEVICE)
    cond = make_cond9(sf_out, mask, dist, copy_resid, scale)

    loss, _ = rectified_flow_loss_scaled(model, r, cond, scale)
    assert torch.isfinite(loss), "scale=0 でも有限な損失を返すこと"


# --- 7. 既存関数の回帰確認 (G2 追加で挙動が変わっていないこと) ------------------

def test_make_cond_unchanged():
    n, hw = 2, 16
    sf_out = torch.arange(n * 3 * hw * hw, dtype=torch.float32).reshape(n, 3, hw, hw)
    mask = torch.full((n, 1, hw, hw), 7.0)
    dist = torch.full((n, 1, hw, hw), 16.0)
    copy_resid = torch.full((n, 3, hw, hw), -3.0)
    cond = make_cond(sf_out, mask, dist, copy_resid)
    assert cond.shape == (n, 8, hw, hw)
    assert torch.equal(cond[:, 0:3], sf_out)
    assert torch.equal(cond[:, 3:4], mask)
    assert torch.allclose(cond[:, 4:5], torch.full((n, 1, hw, hw), 0.5))
    assert torch.equal(cond[:, 5:8], copy_resid)


def test_rectified_flow_loss_unchanged():
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
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters()
                    if p.grad is not None)
    assert grad_norm > 0.0


def test_sample_residual_unchanged():
    model = StageG().to(DEVICE)
    n, hw = 1, 32
    cond = torch.rand(n, 8, hw, hw, device=DEVICE)
    mask = torch.zeros(n, 1, hw, hw, device=DEVICE)
    mask[:, :, 8:24, 8:24] = 1.0

    out = sample_residual(model, cond, mask, steps=4,
                          generator=torch.Generator().manual_seed(0))
    assert out.shape == (n, 3, hw, hw)
    outside = (mask.expand_as(out) < 0.5)
    assert bool((out[outside] == 0).all())
