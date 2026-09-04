"""G0 — 残差が学習可能な構造を持つかを、モデルを作る前に測る (SPEC R2)。

**なぜ最初にこれを測るか**: Stage G は「決定論的予測 (Stage F) の残差だけを
生成モデルで作る」設計。残差 r = 正解 - StageF出力 が**白色雑音に近ければ、
どんなモデルでも学習できない**ので、実装前に確かめる必要がある。
これまで高価な試行を先に走らせて何度も外したので、今回は最初から順序を守る。

測ること:
  1. 残差の振幅 — 元画像の高周波エネルギーに対してどの程度か。
     小さすぎるなら「ぼけ」の正体は残差ではないことになる。
  2. 残差の空間相関 (自己相関の減衰) — 白色雑音なら1画素で落ちる。
     構造を持つなら数画素にわたって相関が残る。
  3. 残差の動径平均 PSD の傾き — 自然画像のテクスチャは 1/f^a の傾きを持つ。
     傾きが 0 に近ければ白色雑音。
  4. **Stage B が検索した実パッチの高周波と、残差の相関** —
     これが有意なら「検索パッチを条件に与える」設計に根拠がある。
     無相関なら条件付けの意味が無く、設計を変える必要がある。
  5. 穴の縁からの距離ごとの残差振幅 — 奥ほど大きいはず (予測が効かないため)。

  pixi run python eval/residual_stats.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.benchmark import load_shard_tiles                     # noqa: E402
from masks import sample_mask                                   # noqa: E402
from nagi_inpaint2 import stage_b_search as sbs                 # noqa: E402
from nagi_inpaint2.stage_f import StageF                        # noqa: E402


def p(*a):
    print(*a, flush=True)


def radial_psd(x: np.ndarray, nbins: int = 20) -> np.ndarray:
    """動径平均パワースペクトル (平均1に正規化)。eval/metrics.py と同じ流儀。"""
    n = min(x.shape)
    n -= n % 2
    if n < 16:
        return np.zeros(nbins)
    x = x[:n, :n]
    win = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2((x - x.mean()) * win))) ** 2
    cy = cx = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (n / 2)
    out = np.array([P[(r >= i / nbins) & (r < (i + 1) / nbins)].mean()
                    if ((r >= i / nbins) & (r < (i + 1) / nbins)).any() else 0.0
                    for i in range(nbins)])
    s = out.mean()
    return out / s if s > 1e-20 else out


def autocorr_decay(x: np.ndarray, lags=(1, 2, 4, 8)) -> dict:
    """水平方向の自己相関。白色雑音なら lag1 で既に 0 付近まで落ちる。"""
    v = x - x.mean()
    d = float((v * v).mean()) + 1e-12
    return {f"lag{l}": float((v[:, :-l] * v[:, l:]).mean() / d) for l in lags}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="cache/bench_person_val.json")
    ap.add_argument("--stage-f", default="checkpoints/stage_f96/last.pt")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    cases = spec["cases"][:a.cases]
    fck = torch.load(a.stage_f, map_location=a.device)
    base = int(fck.get("base") or fck["model"]["out.weight"].shape[1])
    sf = StageF(base=base).to(a.device)
    sf.load_state_dict(fck["model"])
    sf.eval()
    p(f"Stage F: {a.stage_f} (base {base}, step {fck.get('step')})")

    by_shard = {}
    for c in cases:
        by_shard.setdefault(c["shard"], []).append(c)

    amps, hf_amps, psds, acs = [], [], [], []
    corr_copy, corr_rand = [], []
    dist_bins = {b: [] for b in (0, 4, 8, 16, 32)}

    for shard, cs in by_shard.items():
        tiles = {k: im for k, im, _, _ in
                 load_shard_tiles(os.path.join("shards", shard), stride=1)}
        for c in cs:
            if c["key"] not in tiles:
                continue
            img = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(a.device)
            mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(a.device)
            with torch.inference_mode():
                pred = sf(x, mk)
                # Stage B の検索結果 (条件付けの候補)
                q = torch.where(mk > 0.5, pred, x)
                idx, offs, _ = sbs.solve_blocks(x, mk, query=q)
                copy = sbs.paste(x, mk, idx, offs)
                dist = sbs.hole_distance(mk)

            r = (x - pred)[0].permute(1, 2, 0).cpu().numpy()      # 残差
            cp = (copy - pred)[0].permute(1, 2, 0).cpu().numpy()  # 検索パッチの残差
            d = dist[0, 0].cpu().numpy()
            g = img.mean(2)
            m = mask

            # 1. 振幅: 残差 vs 元画像の高周波
            hp = g - F.avg_pool2d(torch.from_numpy(g[None, None]), 5, 1, 2)[0, 0].numpy()
            amps.append(float(np.abs(r.mean(2)[m]).mean()))
            hf_amps.append(float(np.abs(hp[m]).mean()))

            # 2,3. 穴の中の残差の自己相関と PSD (外接矩形で切り出す)
            ys, xs = np.where(m)
            if len(ys) > 64:
                sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                sub = r.mean(2)[sl]
                if min(sub.shape) >= 16:
                    acs.append(autocorr_decay(sub))
                    psds.append(radial_psd(sub))

            # 4. 検索パッチとの相関 (穴の中のみ)
            rv, cv = r.mean(2)[m], cp.mean(2)[m]
            if rv.std() > 1e-8 and cv.std() > 1e-8:
                corr_copy.append(float(np.corrcoef(rv, cv)[0, 1]))
                rnd = np.random.default_rng(0).permutation(cv)
                corr_rand.append(float(np.corrcoef(rv, rnd)[0, 1]))

            # 5. 縁からの距離ごとの振幅
            for b in dist_bins:
                sel = m & (d >= b) & (d < b + 4 if b < 32 else True)
                if sel.sum() > 32:
                    dist_bins[b].append(float(np.abs(r.mean(2)[sel]).mean()))
            if a.device == "mps":
                torch.mps.empty_cache()

    p(f"\nケース {len(amps)} 件\n")
    p("=== 1. 残差の振幅 ===")
    p(f"  残差 |r| 平均              {np.mean(amps):.5f}")
    p(f"  元画像の高周波 |hp| 平均   {np.mean(hf_amps):.5f}")
    p(f"  比 (残差/高周波)           {np.mean(amps)/max(np.mean(hf_amps),1e-9):.3f}")
    p("  -> 比が 1 前後なら、失われた高周波がそのまま残差になっている")

    p("\n=== 2. 残差の空間自己相関 (白色雑音なら lag1 で 0 付近) ===")
    for lag in ("lag1", "lag2", "lag4", "lag8"):
        p(f"  {lag}  {np.mean([d[lag] for d in acs]):+.4f}")

    p("\n=== 3. 残差の動径平均 PSD (低周波→高周波) ===")
    if psds:
        m_psd = np.mean(psds, axis=0)
        p("  " + " ".join(f"{v:.2f}" for v in m_psd[:12]))
        lo, hi = m_psd[:4].mean(), m_psd[-6:].mean()
        p(f"  低域平均 {lo:.3f} / 高域平均 {hi:.3f} / 比 {lo/max(hi,1e-9):.2f}")
        p("  -> 比が 1 に近ければ白色雑音。大きいほど構造がある")

    p("\n=== 4. Stage B の検索パッチとの相関 (条件付けの根拠) ===")
    p(f"  corr(残差, 検索パッチの残差)  {np.mean(corr_copy):+.4f}"
      f"  (中央値 {np.median(corr_copy):+.4f})")
    p(f"  corr(残差, 無作為に並べ替え)  {np.mean(corr_rand):+.4f}   <- 対照")
    p("  -> 対照より明確に大きければ、検索パッチを条件に与える意味がある")

    p("\n=== 5. 穴の縁からの距離ごとの残差振幅 ===")
    for b in sorted(dist_bins):
        v = dist_bins[b]
        if v:
            p(f"  距離 {b:>2}-{b+4 if b < 32 else 999:>3} px   |r| {np.mean(v):.5f}  (n={len(v)})")

    p("\n=== 判定 ===")
    ok_struct = np.mean([d["lag1"] for d in acs]) > 0.3
    ok_cond = np.mean(corr_copy) > np.mean(corr_rand) + 0.05
    p(f"  残差に空間構造がある : {'はい' if ok_struct else 'いいえ'}"
      f"  (lag1 = {np.mean([d['lag1'] for d in acs]):+.3f})")
    p(f"  検索パッチに情報がある: {'はい' if ok_cond else 'いいえ'}")
    if ok_struct:
        p("  -> 残差は学習可能な構造を持つ。Stage G に進んでよい")
    else:
        p("  -> 残差が白色雑音に近い。**Stage G の前提が成立しない**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
