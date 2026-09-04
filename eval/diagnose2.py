"""P0b — ゲート信号の設計を実測で決める (val 上・R1 準拠)。

P0 の発見:
  - このベンチの神託32上限は 20.20 dB (LaMa+1.6)。余白が薄い
  - 大敗ケースは cmin が**低い** (0.0032 vs 全体 0.0119) = 自信を持って間違える。
    ぼけたクエリに対しては滑らかなソースほどコストが下がる系統的バイアスがあるため、
    conf=exp(-cmin) のゲートでは大敗を防げない

そこで val 24ケースで、ブロックごとに以下を集め、
「コピーの真の誤差」を GT なしで予測できる信号を探す:

  cmin  : query 基準の照合コスト最小値
  delta : 低周波不一致 = ||box(copy) - box(query)||² のブロック平均。
          コピー結果が Stage A の構造予測から外れている度合い。推論時に計算可能

方策の比較 (ブロック単位で copy / query を選ぶ):
  A 全コピー (現状)
  B 全クエリ (Stage A のみ)
  C gate-oracle: 真の誤差で copy/query を選ぶ ← P1 の上限
  D full oracle offsets + query   ← P1+P3 の上限
  E cmin 閾値 / F delta 閾値 / G 両方 — グリッドで val 最良を探す

  pixi run python eval/diagnose2.py
"""
from __future__ import annotations

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
from nagi_inpaint2.stage_a import StageA, make_query            # noqa: E402
from nagi_inpaint2.stage_b_simple import _shift                 # noqa: E402

DEV = "mps"
STAGE_A = "checkpoints/stage_a20k/last.pt"
BLOCK = 32
N_CASES = 24


def p(*a):
    print(*a, flush=True)


def block_err(x, img, mask, block=BLOCK):
    """(1,3,H,W) 差の二乗をマスク内で集計したブロック平均 (1,1,h,w)。"""
    e = ((x - img) ** 2).mean(1, keepdim=True) * mask
    den = F.avg_pool2d(mask, block)
    return F.avg_pool2d(e, block) / (den + 1e-8), den


def psnr_from_blocks(err_blk, cov, area_px):
    """ブロック誤差の被覆重み付き平均から PSNR を出す。"""
    tot = float((err_blk * cov).sum() / (cov.sum() + 1e-9))
    return float(10 * np.log10(1.0 / max(tot, 1e-12)))


def main():
    spec = json.load(open("cache/benchmark_val.json"))
    cases = spec["cases"][:N_CASES]
    sa = StageA().to(DEV)
    sa.load_state_dict(torch.load(STAGE_A, map_location=DEV)["model"])
    sa.eval()

    by_shard = {}
    for c in cases:
        by_shard.setdefault(c["shard"], []).append(c)

    rows = []          # per-block: err_copy, err_query, err_oracle, cmin, delta, cov
    case_ids = []
    ci = 0
    for shard, cs in by_shard.items():
        tiles = {k: (im, sm) for k, im, sm, _ in
                 load_shard_tiles(os.path.join("shards", shard), stride=1)}
        for c in cs:
            if c["key"] not in tiles:
                continue
            img, _ = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(DEV)
            mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(DEV)
            with torch.inference_mode():
                q = make_query(sa, x, mk)
                idx, offs, Cn = sbs.solve_blocks(x, mk, query=q)
                copy = sbs.paste(x, mk, idx, offs)
                # 真の誤差 (GT=x はマスク内も本物なので使える; 評価目的)
                e_copy, cov = block_err(copy, x, mk)
                e_query, _ = block_err(q, x, mk)
                # full-oracle offsets
                blk = []
                src = 1.0 - mk
                for oy, ox in offs:
                    cnd = _shift(x, oy, ox)
                    vs = _shift(src, oy, ox)
                    e = ((cnd - x) ** 2).mean(1, keepdim=True)
                    e = torch.where(vs > 0.5, e, torch.full_like(e, 1e6))
                    blk.append(F.avg_pool2d(e * mk, BLOCK))
                e_orc = torch.cat(blk, 1).min(1, keepdim=True).values / (
                    F.avg_pool2d(mk, BLOCK) + 1e-8)
                # 信号
                cmin = Cn.min(1, keepdim=True).values
                lo = lambda t: F.avg_pool2d(t, 8)
                dmap = ((lo(copy) - lo(q)) ** 2).mean(1, keepdim=True)
                delta = F.avg_pool2d(dmap, BLOCK // 8)
                inb = cov[0, 0] > 0.05
                for t in zip(e_copy[0, 0][inb].tolist(), e_query[0, 0][inb].tolist(),
                             e_orc[0, 0][inb].tolist(), cmin[0, 0][inb].tolist(),
                             delta[0, 0][inb].tolist(), cov[0, 0][inb].tolist()):
                    rows.append(t)
                    case_ids.append(ci)
            ci += 1
            torch.mps.empty_cache()

    R = np.array(rows)          # (N,6)
    cid = np.array(case_ids)
    ec, eq, eo, cm, dl, cv = R.T
    p(f"ブロック標本 {len(R):,} (val {ci} ケース)\n")

    def policy_psnr(use_query_mask):
        """ブロックごとに query を使うか。ケースごとに PSNR を出し中央値。"""
        vals = []
        for i in range(ci):
            s = cid == i
            if not s.any():
                continue
            e = np.where(use_query_mask[s], eq[s], ec[s])
            m = float((e * cv[s]).sum() / cv[s].sum())
            vals.append(10 * np.log10(1.0 / max(m, 1e-12)))
        return float(np.median(vals))

    p("=== 方策の上限比較 (val 中央値 PSNR, ブロック集計ベース) ===")
    p(f"  A 全コピー (現状)        {policy_psnr(np.zeros(len(R), bool)):6.2f} dB")
    p(f"  B 全クエリ (StageAのみ)  {policy_psnr(np.ones(len(R), bool)):6.2f} dB")
    p(f"  C gate-oracle           {policy_psnr(eq < ec):6.2f} dB   <- P1 の上限")
    # D: oracle offsets + query
    vals = []
    for i in range(ci):
        s = cid == i
        e = np.minimum(eo[s], eq[s])
        m = float((e * cv[s]).sum() / cv[s].sum())
        vals.append(10 * np.log10(1.0 / max(m, 1e-12)))
    p(f"  D full-oracle+query     {float(np.median(vals)):6.2f} dB   <- P1+P3 の上限")

    p("\n=== 信号と真のコピー誤差の相関 (log 空間) ===")
    lg = lambda v: np.log10(np.maximum(v, 1e-9))
    p(f"  corr(log err_copy, log cmin) : {np.corrcoef(lg(ec), lg(cm))[0,1]:+.3f}")
    p(f"  corr(log err_copy, log delta): {np.corrcoef(lg(ec), lg(dl))[0,1]:+.3f}")
    p(f"  corr(log err_copy, log cmin+delta 和): "
      f"{np.corrcoef(lg(ec), lg(cm + dl))[0,1]:+.3f}")

    p("\n=== 閾値方策のグリッド (query へ退避する条件 / val 中央値 PSNR) ===")
    best = None
    for name, sig in (("cmin", cm), ("delta", dl)):
        for qt in (50, 60, 70, 75, 80, 85, 90, 95):
            t = np.percentile(sig, qt)
            v = policy_psnr(sig > t)
            if best is None or v > best[0]:
                best = (v, f"{name}>p{qt}")
            p(f"  {name:>5} > p{qt:<3}  {v:6.2f} dB")
    # 予測比較方策: eq_hat vs ec_hat。ec の予測として α·delta を使う
    for alpha in (0.5, 1.0, 2.0, 4.0, 8.0):
        v = policy_psnr(alpha * dl > np.maximum(eq * 0 + np.median(eq), 1e-9) * 0 + eq)
        # ↑ 「α·delta が query の真の誤差を超えるなら退避」は eq を使うためオラクル寄り。
        # 推論可能版: eq の代わりに定数 (中央値) を使う
    med_eq = np.median(eq)
    for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
        v = policy_psnr(alpha * dl > med_eq)
        if v > best[0]:
            best = (v, f"alpha*delta>med_eq (a={alpha})")
        p(f"  {alpha}*delta > const     {v:6.2f} dB")
    p(f"\n  最良方策: {best[1]}  {best[0]:.2f} dB")


if __name__ == "__main__":
    main()
