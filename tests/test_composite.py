"""信頼度ゲート合成 (`block_confidence` / `composite` / `inpaint(fallback=True)`) の回帰テスト。

device は "cpu" 固定 (mps は他ジョブが使用中のため使わない)。

**`block_confidence` の median は自己参照的である。** `s` は `Cn` 全体
(= 判定対象のブロック自身を含む集合) の中央値なので、対象ブロックの
コストがちょうど `s` と一致するとき（同質な母集団ならほぼ必ずそうなる）
`conf = exp(-1/tau_c)` に張り付く。`tau_c=1.0` ではこれが 0.368 であり、
**母集団が同質な限り conf の中央値は 0.5 を超えない**
(`median(f(X)) = f(median(X))`, f は単調減少なので常に成立する不等式)。
テスト1 で `tau_c` を既定値より大きく取っているのはこのため
(数式的に必要な下限は `tau_c > 1/ln2 ≈ 1.443`)。マスクをブロック格子の
全域に広げているのは、マスク外のブロックが `Cn=0` に潰れて中央値を
汚染する (穴の外は `c*mask` で常に 0 になる) のを避けるため — この汚染は
下方向にしか効かないので、外側ブロックが多いテスト1のような小さな穴では
`s<=0` の安全側分岐に落ちて confidence が全ゼロになってしまう
(実際に 160x160 中央28x28 マスクで検証して確認した)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_b_search import (   # noqa: E402
    block_confidence, composite, inpaint, paste, solve_blocks,
)

DEVICE = "cpu"


def _periodic_texture(n: int, noise: float, seed: int) -> torch.Tensor:
    """x 方向にのみ周期的な (= y 方向には不変な) テクスチャ。

    y 方向のどんなシフトも厳密に一致するので、穴の中身と一致する参照元が
    必ず存在する ("真の一致が存在する" ケースの構成)。
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


# --- テスト1: 周期テクスチャ (真の一致が存在する) --------------------------

# マスクをブロック (既定 32px) の境界をまたぐように置き、4ブロック全てに
# 穴がかかるようにする。こうしないとマスク外のブロックの Cn=0 が中央値を
# 汚染し、s<=0 の安全側分岐に落ちて conf が全ゼロになる (上の docstring 参照)。
_N1, _LO1, _HI1 = 64, 18, 46
_TAU_HIGH = 2.5   # exp(-1/2.5) ≈ 0.67 > 0.5 の余裕を持たせる


def test_block_confidence_high_for_true_match():
    img = _periodic_texture(_N1, noise=0.01, seed=0)
    mask = _center_mask(_N1, _LO1, _HI1)
    q = _blur_query(img)

    _, _, Cn = solve_blocks(img, mask, query=q)
    conf = block_confidence(Cn, tau_c=_TAU_HIGH)

    mask_ds = F.avg_pool2d(mask, 32) > 0
    vals = conf[0, 0][mask_ds[0, 0]]
    assert vals.numel() > 0
    assert vals.median().item() > 0.5


def test_fallback_close_to_no_fallback_when_match_exists():
    """真の一致があるとき、fallback=True は fallback=False (同じ query) に近い。

    比較対象は同じ query を渡した fallback=False (= paste そのもの)。
    query の有無で idx の解が変わってしまうため、query なしの `inpaint(img, mask)`
    と比べると (fallback 機構と無関係に) 大きく食い違う場合がある。
    """
    img = _periodic_texture(_N1, noise=0.01, seed=0)
    mask = _center_mask(_N1, _LO1, _HI1)
    q = _blur_query(img)

    out_fb = inpaint(img, mask, fallback=True, tau_c=_TAU_HIGH, query=q)
    out_plain = inpaint(img, mask, query=q)   # fallback=False (既定)

    m = mask.expand_as(img) > 0.5
    diff = (out_fb - out_plain).abs()[m].mean().item()
    assert diff < 0.05


# --- テスト2: 一意なコンテンツ (一致が存在しない) ---------------------------

_N2, _LO2, _HI2, _SEED2 = 64, 18, 46, 0


def test_block_confidence_low_for_unique_content():
    rng = np.random.default_rng(_SEED2)
    img_np = rng.uniform(0, 1, (_N2, _N2, 3)).astype(np.float32)
    img = torch.from_numpy(img_np.transpose(2, 0, 1)[None]).to(DEVICE)
    mask = _center_mask(_N2, _LO2, _HI2)
    q = _blur_query(img)

    _, _, Cn = solve_blocks(img, mask, query=q)
    conf = block_confidence(Cn)   # 既定 tau_c=1.0

    mask_ds = F.avg_pool2d(mask, 32) > 0
    vals = conf[0, 0][mask_ds[0, 0]]
    assert vals.numel() > 0
    assert vals.median().item() < 0.5


def test_fallback_output_closer_to_query_when_no_match():
    rng = np.random.default_rng(_SEED2)
    img_np = rng.uniform(0, 1, (_N2, _N2, 3)).astype(np.float32)
    img = torch.from_numpy(img_np.transpose(2, 0, 1)[None]).to(DEVICE)
    mask = _center_mask(_N2, _LO2, _HI2)
    q = _blur_query(img)

    idx, offs, Cn = solve_blocks(img, mask, query=q)
    copy = paste(img, mask, idx, offs)
    out = inpaint(img, mask, fallback=True, query=q)

    m = mask.expand_as(img) > 0.5
    dist_to_query = (out - q).abs()[m].mean().item()
    dist_to_copy = (out - copy).abs()[m].mean().item()
    assert dist_to_query < dist_to_copy


# --- テスト3: 後方互換 (fallback=False は従来と bit 一致) -------------------

def _reference_inpaint(img, mask, feat=None, **kw):
    """信頼度ゲート導入前の `inpaint` と同じ処理 (solve_blocks -> paste)。"""
    idx, offs, _ = solve_blocks(img, mask, feat, **kw)
    return paste(img, mask, idx, offs)


@pytest.mark.parametrize("with_query", [False, True])
def test_fallback_false_matches_legacy_inpaint(with_query):
    img = _periodic_texture(56, noise=0.02, seed=1)
    mask = _center_mask(56, 20, 36)
    kw = {}
    if with_query:
        kw["query"] = _blur_query(img)

    out_new = inpaint(img, mask, fallback=False, **kw)
    out_ref = _reference_inpaint(img, mask, **kw)
    assert torch.equal(out_new, out_ref)

    # fallback 引数を渡さない (既定 False) 場合も同じであること
    out_default = inpaint(img, mask, **kw)
    assert torch.equal(out_default, out_ref)


def test_fallback_true_without_query_matches_legacy():
    """fallback=True でも query が無ければ従来どおり (query が無いと退避できない)。"""
    img = _periodic_texture(56, noise=0.02, seed=2)
    mask = _center_mask(56, 20, 36)
    out_fb = inpaint(img, mask, fallback=True)
    out_ref = _reference_inpaint(img, mask)
    assert torch.equal(out_fb, out_ref)


# --- テスト4: solve_blocks の戻り値の形・型 ---------------------------------

def test_solve_blocks_return_shapes_and_types():
    n = 48
    img = torch.rand(1, 3, n, n, device=DEVICE)
    mask = _center_mask(n, 16, 32)

    idx, offs, Cn = solve_blocks(img, mask)

    assert idx.dtype == torch.long
    assert Cn.dtype == torch.float32
    assert isinstance(offs, list)
    assert all(isinstance(o, tuple) and len(o) == 2 for o in offs)

    h, w = n // 32, n // 32
    assert idx.shape == (1, 1, h, w)
    assert Cn.shape == (1, len(offs), h, w)
