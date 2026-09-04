"""P0 診断 — 固定テストベンチ上で「神託上限」と「敗因」を測る (SPEC §13.3)。

答えるべき問い:
  Q1 このベンチ (保留shard・難しい内容) での神託上限はいくつか。
     headroom が無ければ以降の全フェーズが無意味 (R2)。
  Q2 LaMa に負けるケースはどこか。照合コスト (query 基準の cmin) と
     相関するなら、P1 の信頼度ゲート合成が正しい処方になる。

出力: cache/diag_test.json (ケースごとの LaMa / 我々 / 神託32 の PSNR と照合統計)

  pixi run python eval/diagnose_bench.py
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
from eval.lama_baseline import Lama                             # noqa: E402
from masks import sample_mask                                   # noqa: E402
from nagi_inpaint2 import stage_b_search as sbs                 # noqa: E402
from nagi_inpaint2 import stage_d                               # noqa: E402
from nagi_inpaint2.stage_a import StageA, make_query            # noqa: E402
from nagi_inpaint2.stage_b_simple import _shift                 # noqa: E402

DEV = "mps"
STAGE_A = "checkpoints/stage_a20k/last.pt"


def p(*a):
    print(*a, flush=True)


def psnr(out, img, mask):
    sel = mask[..., None].repeat(3, axis=2) if out.ndim == 3 else mask
    d = ((out - img) ** 2)[sel]
    m = float(d.mean())
    return float("inf") if m <= 0 else float(10 * np.log10(1.0 / m))


def oracle_block(img_t, mask_t, block=32, radius=96, stride=4):
    """GT を見てブロックごとに最良オフセットを選ぶ (二段パスでメモリを抑える)。"""
    src = 1.0 - mask_t
    offs = sbs._offsets(radius, stride)
    blk, tot = [], []
    for oy, ox in offs:
        c = _shift(img_t, oy, ox)
        vs = _shift(src, oy, ox)
        e = ((c - img_t) ** 2).mean(1, keepdim=True)
        e = torch.where(vs > 0.5, e, torch.full_like(e, 1e6))
        blk.append(F.avg_pool2d(e * mask_t, block))
    j = torch.cat(blk, 1).argmin(1, keepdim=True)
    jf = F.interpolate(j.float(), size=img_t.shape[-2:], mode="nearest").long()[0, 0]
    out = img_t.clone()
    for i in torch.unique(jf).tolist():
        oy, ox = offs[int(i)]
        m = (jf == i)[None, None].expand_as(img_t)
        out = torch.where(m, _shift(img_t, oy, ox), out)
    return out


def main():
    spec = json.load(open("cache/benchmark.json"))
    sa = StageA().to(DEV)
    sa.load_state_dict(torch.load(STAGE_A, map_location=DEV)["model"])
    sa.eval()
    lama = Lama(device=DEV)

    by_shard = {}
    for c in spec["cases"]:
        by_shard.setdefault(c["shard"], []).append(c)

    rows = []
    for shard, cases in by_shard.items():
        tiles = {k: (im, sm) for k, im, sm, _ in
                 load_shard_tiles(os.path.join("shards", shard), stride=1)}
        for c in cases:
            if c["key"] not in tiles:
                continue
            img, satm = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(DEV)
            mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(DEV)

            fl = lama(img, mask)
            p_lama = psnr(np.where(mask[..., None], fl, img), img, mask)

            with torch.inference_mode():
                q = make_query(sa, x, mk)
                idx, offs, Cn = sbs.solve_blocks(x, mk, query=q)
                o = sbs.paste(x, mk, idx, offs)[0].permute(1, 2, 0).cpu().numpy()
            d = stage_d.harmonize(o, img, mask, valid=~satm,
                                  rng=np.random.default_rng(0),
                                  do_color=False, do_poisson=True, do_grain=False)
            p_ours = psnr(np.where(mask[..., None], d, img), img, mask)

            with torch.inference_mode():
                orc = oracle_block(x, mk)[0].permute(1, 2, 0).cpu().numpy()
            p_orc = psnr(np.where(mask[..., None], orc, img), img, mask)

            # 照合統計: マスクを含むブロックの cmin (query 基準)
            cov = F.avg_pool2d(mk, 32)[0, 0]
            cmin = Cn.min(1).values[0]
            inb = cov > 0.1
            stats = dict(
                cmin_med=float(cmin[inb].median()) if inb.any() else float("nan"),
                cmin_p90=float(cmin[inb].quantile(0.9)) if inb.any() else float("nan"),
                nblocks=int(inb.sum()))
            rows.append(dict(shard=shard, key=c["key"], area=c["area"],
                             hf=c["hf"], lama=p_lama, ours=p_ours,
                             oracle=p_orc, **stats))
            torch.mps.empty_cache()

    json.dump(rows, open("cache/diag_test.json", "w"), indent=1)
    L = np.array([r["lama"] for r in rows])
    O = np.array([r["ours"] for r in rows])
    R = np.array([r["oracle"] for r in rows])
    C = np.array([r["cmin_med"] for r in rows])
    diff = O - L
    p(f"\nケース {len(rows)} 件")
    p(f"  LaMa   中央値 {np.median(L):6.2f} dB")
    p(f"  我々   中央値 {np.median(O):6.2f} dB  (勝率 {(diff>0).sum()}/{len(rows)})")
    p(f"  神託32 中央値 {np.median(R):6.2f} dB  <- このベンチの上限")
    p(f"  headroom (神託-我々) 中央値 {np.median(R-O):6.2f} dB / p25 {np.percentile(R-O,25):.2f} / p75 {np.percentile(R-O,75):.2f}")
    ok = np.isfinite(C)
    p(f"\n  相関 (我々-LaMa) vs cmin_med : r={np.corrcoef(C[ok], diff[ok])[0,1]:+.3f}")
    p(f"  相関 (我々-LaMa) vs マスク面積: r={np.corrcoef([r['area'] for r in rows], diff)[0,1]:+.3f}")
    p(f"  相関 (我々-LaMa) vs 高周波比  : r={np.corrcoef([r['hf'] for r in rows], diff)[0,1]:+.3f}")
    # 負けが大きい10ケースの cmin
    worst = np.argsort(diff)[:10]
    p(f"\n  大敗10ケースの cmin_med 中央値: {np.median(C[worst]):.4f}  全体: {np.median(C[ok]):.4f}")
    p(f"  -> 大敗ケースの cmin が高ければ、P1 (信頼度ゲート) が正しい処方")


if __name__ == "__main__":
    main()
