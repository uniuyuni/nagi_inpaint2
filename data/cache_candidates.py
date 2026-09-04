"""照合コスト学習用のキャッシュを作る (SPEC §4.2)。

**学習の対象を明確にする**

実測で分かった構図:

    神託 (正解を見てブロックごとに最良オフセットを選ぶ)   33.27 dB
    実測 (周囲の既知画素の照合コストで選ぶ)              23.88 dB
    LaMa                                            29.53 dB

差の 9.4 dB は**「正解なしでどのオフセットが良いかを当てる精度」**である。
これを縮めるのが FeatureNet の唯一の仕事。探索そのものは学習対象ではない。

**なぜキャッシュするか**

前回の学習は 1 step 4.8秒だった。毎ステップ 2400候補の全探索を
`no_grad` でやり直しており、勾配に寄与しない計算に時間の大半を使っていた。

ランキングを学ぶのに 2400候補すべては要らない。**誤差の範囲を張る少数の候補**
があれば十分。ここでは 1タイルあたり N_SAMPLE 個のオフセットをランダムに取り、
そのブロックごとの実誤差を事前計算しておく。実誤差は特徴に依存しないので、
一度計算すれば学習中は使い回せる。

学習時はキャッシュされた少数オフセットについてのみコストを計算するので、
1 step が 2桁速くなる。

出力: cache/match_train.npz

  pixi run python data/cache_candidates.py --tiles 1500
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from masks import sample_mask                                    # noqa: E402
from nagi_inpaint2.color import byte_unshuffle, dequantize       # noqa: E402
from nagi_inpaint2.stage_b_search import _offsets, _shift        # noqa: E402

OUT = "cache/match_train.npz"
RADIUS, STRIDE, BLOCK = 96, 4, 32
N_SAMPLE = 64          # 1タイルあたりに実誤差を測るオフセット数


def p(*a):
    print(*a, flush=True)


def iter_tiles(shard_dir: str, n: int, seed: int = 0):
    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".tar"))
    rng = np.random.default_rng(seed)
    pick = shards[::max(1, len(shards) // 60)][:60]
    per = max(1, n // len(pick))
    d = zstd.ZstdDecompressor()
    got = 0
    for s in pick:
        with tarfile.open(os.path.join(shard_dir, s)) as tf:
            keys = sorted(m[:-5] for m in tf.getnames() if m.endswith(".json"))
            if not keys:
                continue
            for k in rng.choice(keys, size=min(per, len(keys)), replace=False):
                meta = json.loads(tf.extractfile(k + ".json").read())
                img = byte_unshuffle(
                    d.decompress(tf.extractfile(k + ".img.u16.zst").read()),
                    tuple(meta["shape"]))
                yield dequantize(img, meta["q_scale"], meta["q_offset"]), meta
                got += 1
                if got >= n:
                    return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--tiles", type=int, default=1500)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mask-kind", default="blob")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    offs_all = _offsets(RADIUS, STRIDE)
    rng = np.random.default_rng(a.seed)
    imgs, masks, off_idx, errs, covs, vfracs = [], [], [], [], [], []
    t0 = time.perf_counter()
    n = 0

    for p_img, meta in iter_tiles(a.shards, a.tiles, a.seed):
        mask_np, _ = sample_mask(meta["tile"], rng, kind=a.mask_kind)
        if mask_np.sum() < 2000:
            continue
        img = torch.from_numpy(p_img.transpose(2, 0, 1)[None]).to(a.device)
        mask = torch.from_numpy(mask_np[None, None].astype(np.float32)).to(a.device)
        src = 1.0 - mask

        sel = rng.choice(len(offs_all), size=N_SAMPLE, replace=False)
        with torch.no_grad():
            # **マスク被覆で正規化する。** そのまま avg_pool すると、マスク外の
            # ブロック (タイルの大半) が 0 になり、統計も損失も無意味になる。
            cov = F.avg_pool2d(mask, BLOCK)               # ブロックのマスク被覆率
            e, vfrac = [], []
            for i in sel:
                oy, ox = offs_all[int(i)]
                c = _shift(img, oy, ox)
                vs = _shift(src, oy, ox)                  # 参照元が穴の外か
                # **有効性を誤差に混ぜない。** ペナルティ値を誤差に入れると
                # 統計も損失も無効オフセットに支配される (実測で誤差 p90 が 976 になった)。
                # 誤差は「マスク内かつ参照元が有効」な画素だけで測り、
                # 有効割合は別に記録して、学習側で候補の採否に使う。
                use = mask * vs
                d2 = ((c - img) ** 2).mean(1, keepdim=True) * use
                num = F.avg_pool2d(d2, BLOCK)
                den = F.avg_pool2d(use, BLOCK)
                e.append(num / (den + 1e-8))
                vfrac.append(den / (cov + 1e-8))
            E = torch.cat(e, 1)[0].cpu().numpy()          # (N_SAMPLE, h/B, w/B)
            V = torch.cat(vfrac, 1)[0].cpu().numpy()
            COV = cov[0, 0].cpu().numpy()

        imgs.append((p_img * 4096).astype(np.int16))       # 保存を軽くする
        masks.append(np.packbits(mask_np))
        off_idx.append(sel.astype(np.int16))
        errs.append(E.astype(np.float32))
        covs.append(COV.astype(np.float32))
        vfracs.append(V.astype(np.float32))
        n += 1
        if n % 50 == 0:
            el = time.perf_counter() - t0
            p(f"  {n}/{a.tiles}  {el/n:.2f}s/タイル  "
              f"残り約 {el/n*(a.tiles-n)/60:.0f}分")

    if not imgs:
        p("タイルが集まらなかった")
        return 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out, imgs=np.stack(imgs), masks=np.stack(masks),
        off_idx=np.stack(off_idx), errs=np.stack(errs),
        covs=np.stack(covs), vfracs=np.stack(vfracs),
        offsets=np.array(offs_all, dtype=np.int16),
        block=BLOCK, radius=RADIUS, stride=STRIDE, img_scale=4096)
    p(f"\n{n} タイル -> {a.out} ({os.path.getsize(a.out)/1e6:.0f} MB)")
    E, C, V = np.stack(errs), np.stack(covs), np.stack(vfracs)
    blk = C > 0.25                                    # マスクを十分含むブロック
    use = blk[:, None] & (V > 0.9)                    # かつ候補が有効
    p(f"対象ブロック (マスク被覆 >25%): {int(blk.sum()):,} / {C.size:,}")
    p(f"  1ブロックあたり有効候補数 中央値 {np.median(use.sum(1)[blk]):.0f}/{N_SAMPLE}")
    fin = E[use]
    p(f"実誤差: p10 {np.percentile(fin,10):.5f} p50 {np.percentile(fin,50):.5f} "
      f"p90 {np.percentile(fin,90):.5f}")
    Em = np.where(use, E, np.nan)
    with np.errstate(all="ignore"):
        best = np.nanmin(Em, 1); worst = np.nanmax(Em, 1)
    ok = np.isfinite(best) & np.isfinite(worst) & (use.sum(1) >= 4)
    p(f"候補間の誤差比 (最悪/最良): p25 {np.percentile((worst/np.maximum(best,1e-9))[ok],25):.1f} "
      f"p50 {np.percentile((worst/np.maximum(best,1e-9))[ok],50):.1f}")
    p(f"  -> この比が大きいほど「正しく選ぶ」価値がある")
    return 0


if __name__ == "__main__":
    sys.exit(main())
