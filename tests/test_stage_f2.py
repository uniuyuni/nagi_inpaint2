"""Stage F 追加学習 (`train/train_stage_f2.py`, 知覚損失 + 敵対的損失) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない。学習は別途行う)。

確認すること:
  1. PatchGAN の forward が通り、出力が (N,1,h,w) の形であること
  2. 識別器に BatchNorm が含まれないこと (マスク入力で統計が壊れた過去の失敗を
     StageF 本体だけでなく識別器でも再発させない)
  3. 知覚損失 (AlexNet) が有限で、同一画像なら 0 に近いこと
  4. **知覚損失がぼけを検出すること** (最重要。これが導入理由そのもの):
     元画像とその強いぼかしの知覚損失が、元画像と弱いぼかしの知覚損失より大きい。
  5. 生成器の合計損失が backward でき、生成器パラメータに非ゼロ勾配が流れること
  6. `--lam-adv 0` 相当のとき識別器が呼ばれず、損失に adv 項が入らないこと
     (関数レベル: `step_loss_gen` に `lam_adv=0.0` を渡して確認)

AlexNet は `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` の既存
キャッシュから読む (新規ダウンロードなし)。`AlexNetPerceptual` の構築コストが
高いのでモジュール scope の fixture で使い回す (`test_perceptual.py` と同じ方針)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_f import StageF                     # noqa: E402
from train.train_stage_f2 import (                           # noqa: E402
    AlexNetPerceptual,
    PatchGAN,
    step_loss_gen,
)

DEVICE = "cpu"
N = 128


def _sample(n: int, batch: int = 1, seed: int = 0):
    """`test_stage_f.py._sample` と同じ流儀: 穴が空にならない程度の割合で作る。"""
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(batch, 3, n, n, generator=g, device=DEVICE)
    mask = (torch.rand(batch, 1, n, n, generator=g, device=DEVICE) > 0.7).float()
    return img, mask


def _photo_like(n: int, seed: int) -> np.ndarray:
    """`test_perceptual.py._photo_like` と同じ合成テクスチャ (自然画像に近い
    構造+微細ノイズ)。値域はこのリポジトリの知覚空間の慣習 (0..1.2 前後)。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    base = (0.5
            + 0.2 * np.sin(xx / 5.0)
            + 0.2 * np.cos(yy / 7.0)
            + 0.1 * np.sin((xx + yy) / 3.0))
    img = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.2).astype(np.float32)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        out[..., c] = gaussian_filter(x[..., c], sigma=sigma)
    return out


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    """(H,W,3) numpy -> (1,3,H,W) torch tensor。"""
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(DEVICE)


@pytest.fixture(scope="module")
def perceptual() -> AlexNetPerceptual:
    return AlexNetPerceptual(device=DEVICE)


# --- 1. PatchGAN の forward 形状 ---------------------------------------------

def test_patchgan_forward_shape():
    disc = PatchGAN().to(DEVICE)
    img = torch.rand(2, 3, 256, 256, device=DEVICE)
    mask = torch.rand(2, 1, 256, 256, device=DEVICE)
    with torch.no_grad():
        out = disc(img, mask)
    assert out.dim() == 4
    assert out.shape[0] == 2
    assert out.shape[1] == 1
    assert out.shape[2] > 1 and out.shape[3] > 1


# --- 2. BatchNorm を含まないこと --------------------------------------------

def test_patchgan_no_batchnorm():
    disc = PatchGAN()
    bn = [name for name, m in disc.named_modules()
          if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.BatchNorm3d, nn.SyncBatchNorm))]
    assert bn == [], f"BatchNorm を含んでいる: {bn}"


# --- 3. 同一画像の知覚損失は 0 に近い ------------------------------------------

def test_perceptual_identical_images_near_zero(perceptual):
    x_np = _photo_like(N, seed=0)
    x = _to_tensor(x_np)
    mask = torch.ones(1, 1, N, N, device=DEVICE)
    d = perceptual(x, x, mask)
    assert torch.isfinite(d)
    assert float(d) < 1e-5


# --- 4. ぼかしを検出できること (最重要、導入理由そのもの) -----------------------

def test_perceptual_detects_blur(perceptual):
    """我々の弱点は「ぼけ」なので、学習の知覚損失がぼけに反応しなければ
    導入した意味がない。強いぼかしのほうが弱いぼかしより距離が大きいことを見る。
    """
    x_np = _photo_like(N, seed=1)
    weak_np = _blur(x_np, sigma=0.8)
    strong_np = _blur(x_np, sigma=3.0)

    x = _to_tensor(x_np)
    weak = _to_tensor(weak_np)
    strong = _to_tensor(strong_np)
    mask = torch.ones(1, 1, N, N, device=DEVICE)

    d_weak = perceptual(x, weak, mask)
    d_strong = perceptual(x, strong, mask)

    print(f"\n[blur] weak(sigma=0.8) -> {float(d_weak):.6f}  "
          f"strong(sigma=3.0) -> {float(d_strong):.6f}")
    assert torch.isfinite(d_weak)
    assert torch.isfinite(d_strong)
    assert float(d_strong) > float(d_weak)


# --- 5. 生成器の合計損失が backward でき、勾配が流れること ----------------------

def test_generator_loss_backward_nonzero_grad(perceptual):
    model = StageF().to(DEVICE)
    img, mask = _sample(64, batch=1, seed=2)

    total, pred, parts = step_loss_gen(
        model, None, img, mask,
        lam_grad=0.5, lam_known=0.3, lam_perc=1.0, lam_adv=0.0,
        perceptual=perceptual)

    assert total is not None
    assert torch.isfinite(total)
    for key in ("l1", "l1k", "grad", "perc", "adv"):
        assert key in parts
        assert np.isfinite(parts[key])

    model.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters()
                    if p.grad is not None)
    assert grad_norm > 0.0, "backward しても勾配が流れていない"


def test_step_loss_gen_none_when_mask_empty(perceptual):
    """穴が無ければ損失を定義できない (StageA/StageF と同じ規約)。"""
    model = StageF().to(DEVICE)
    img = torch.rand(1, 3, 64, 64, device=DEVICE)
    mask = torch.zeros(1, 1, 64, 64, device=DEVICE)
    total, pred, parts = step_loss_gen(
        model, None, img, mask,
        lam_grad=0.5, lam_known=0.3, lam_perc=1.0, lam_adv=0.0,
        perceptual=perceptual)
    assert total is None
    assert pred is None
    assert parts is None


# --- 6. lam_adv=0 のとき識別器が呼ばれないこと ---------------------------------

def test_lam_adv_zero_skips_discriminator(perceptual):
    calls = {"n": 0}

    class _CountingDisc(nn.Module):
        """本物の PatchGAN を包んで forward 呼び出し回数を数えるだけのラッパー。"""

        def __init__(self):
            super().__init__()
            self.inner = PatchGAN()

        def forward(self, img, mask):
            calls["n"] += 1
            return self.inner(img, mask)

    disc = _CountingDisc().to(DEVICE)
    model = StageF().to(DEVICE)
    img, mask = _sample(64, batch=1, seed=3)

    total, pred, parts = step_loss_gen(
        model, disc, img, mask,
        lam_grad=0.5, lam_known=0.3, lam_perc=1.0, lam_adv=0.0,
        perceptual=perceptual)

    assert calls["n"] == 0, "lam_adv=0 なのに識別器が呼ばれた"
    assert total is not None
    assert parts["adv"] == 0.0
