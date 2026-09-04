"""照合コスト学習 v2 用のキャッシュを作る (SPEC §4.2)。

`data/cache_candidates.py` (v1) との違いは2つ:

  1. **query 対応。** `stage_b_search.solve_blocks` は `query` (Stage A の
     低周波補完で穴を埋めた画像) を渡すと全ブロックが照合可能になる
     (照合可能率 36% -> 100%、+8.7 dB — `stage_b_search.py` のコメント参照)。
     v1 は query 非対応で img そのものを照合対象にしていたため、
     推論時 (`solve_blocks(..., query=q)`) と学習時で特徴抽出の入力分布が
     ずれていた。v2 は Stage A で作った query をタイルごとに保存し、
     学習側 (`train/train_matcher2.py`) は常に query 上で特徴を取る。
  2. **保留 shard を混入させない。** `train/dataset.split_shards` の train 側
     shard からのみタイルを取る。保留 shard には無作為抽出でしか触れず、
     混ざると学習が事実上の "answer leak" になる (§6.4 のコメント参照)。

**オフセットのサンプリングを2種類混ぜる。** N_SAMPLE=64 のうち 40 個は
全オフセット (radius 96, stride 4 = 2400候補) から一様ランダムに選ぶが、
残り 24 個は **手作り特徴 + query でのコストが低い上位オフセット**を選ぶ。
一様ランダムだけだと大半が「明らかに悪い」オフセットになり、学習が
「良い候補と悪い候補を見分ける」練習ばかりして「良い候補どうしの
優劣をつける」練習ができない。現状の手作り特徴が競合として選びがちな
オフセット (= 真の正解の周りの hard negative、あるいは正解そのもの) を
混ぜることで、学習信号を難しい所に集中させる。

**選び方はブロックごとのコストではなく「平均順位」で決める。** コストの
絶対値は無効オフセットで BIG (1e6) に張り付くため、単純平均だと少数の
無効ブロックが全体を支配する。順位 (0 が最良) の平均なら BIG の影響が
「最下位」に丸められるだけで済み、頑健に「タイル全体でコンスタントに
良いオフセット」を選べる。

真の誤差 (`errs`)・有効率 (`vfracs`)・被覆 (`covs`) の定義は v1 と同じ
(GT との差、mask∧vs 画素のみで平均、被覆で正規化)。学習の教師信号は
「正解を見た誤差」なので、query ではなく GT (`img`) を使って測る。

出力: cache/match_train2.npz

  pixi run python data/cache_candidates2.py --tiles 1200
"""
from __future__ import annotations

import argparse
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

from masks import sample_mask                                        # noqa: E402
from nagi_inpaint2.color import byte_unshuffle, dequantize           # noqa: E402
from nagi_inpaint2.stage_a import StageA, make_query                 # noqa: E402
from nagi_inpaint2.stage_b_search import (                           # noqa: E402
    BLOCK, MIN_W, PATCH, SEARCH_RADIUS, SEARCH_STRIDE, _offsets, _shift,
)
from nagi_inpaint2.stage_b_simple import _box, _features             # noqa: E402
from train.dataset import list_shards, split_shards                  # noqa: E402

OUT = "cache/match_train2.npz"
RADIUS, STRIDE = SEARCH_RADIUS, SEARCH_STRIDE
N_SAMPLE = 64          # 1タイルあたりに実誤差を測るオフセット数
N_HARD = 24            # うち手作り特徴+queryでの上位 (hard) の個数
N_RAND = N_SAMPLE - N_HARD
BIG = 1e6              # solve_blocks と同じ「無効オフセット」の値


def p(*a):
    print(*a, flush=True)


def iter_tiles(shard_paths: list[str], n: int, seed: int = 0):
    """train 側 shard だけを舐めてタイルを yield する (v1 と同じ間引き方)。"""
    rng = np.random.default_rng(seed)
    pick = shard_paths[::max(1, len(shard_paths) // 60)][:60]
    per = max(1, n // len(pick))
    d = zstd.ZstdDecompressor()
    got = 0
    for s in pick:
        with tarfile.open(s) as tf:
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


def _handcrafted_query_cost(q: torch.Tensor, mask: torch.Tensor, src: torch.Tensor,
                            oy: int, ox: int, r: int) -> torch.Tensor:
    """solve_blocks の query 分岐と同一式で1オフセットぶんのブロックコストを返す。

    query パスでは qval=ones (query は全画素「中身がある」) なので、
    f=_features(q)*1、w=vs (ソース有効性のみ) になる。
    """
    f = _features(q)
    vs = _shift(src, oy, ox)
    w = vs
    wb = _box(w, r)
    c = _box(((f - _shift(f, oy, ox)) ** 2).sum(1, keepdim=True) * w, r) / (wb + 1e-6)
    bad = (vs <= 0.5) | (wb < MIN_W)
    c = torch.where(bad, torch.full_like(c, BIG), c)
    return F.avg_pool2d(c * mask, BLOCK)


def pick_hard_offsets(q: torch.Tensor, mask: torch.Tensor, src: torch.Tensor,
                      offs_all: list[tuple[int, int]], cov: torch.Tensor,
                      r: int, k: int) -> np.ndarray:
    """全オフセットのコストを計算し、「マスクを含むブロックの平均順位」が
    最良の k 個を返す (offs_all 内のインデックス)。
    """
    costs = [_handcrafted_query_cost(q, mask, src, oy, ox, r) for oy, ox in offs_all]
    Cn = torch.cat(costs, 1)[0]                     # (K,h,w)
    blk = cov[0, 0] > 0                              # マスクを含むブロック
    if not bool(blk.any()):
        blk = torch.ones_like(blk)                   # 保険 (呼び出し側で area>=2000 を担保済み)
    Csel = Cn[:, blk]                                # (K,M)
    # 順位 (0=最良) で平均する。BIG の絶対値ではなく「最下位」扱いに丸まるので
    # 一部の無効ブロックに全体が引きずられない。
    rank = Csel.argsort(0).argsort(0).float()
    avg_rank = rank.mean(1)                          # (K,)
    return torch.argsort(avg_rank)[:k].cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--tiles", type=int, default=1200)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mask-kind", default="blob")
    ap.add_argument("--stage-a", default="checkpoints/stage_a20k/last.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    all_shards = list_shards(a.shards)
    train_shards, val_shards = split_shards(all_shards)
    p(f"shard 全 {len(all_shards)} 本 -> train {len(train_shards)} 本を使用 "
      f"(保留 {len(val_shards)} 本は混入させない)")

    sa = StageA().to(a.device)
    ckpt = torch.load(a.stage_a, map_location=a.device)
    sa.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    sa.eval()

    offs_all = _offsets(RADIUS, STRIDE)
    r = PATCH // 2
    rng = np.random.default_rng(a.seed)

    imgs, queries, masks, off_idx = [], [], [], []
    errs, covs, vfracs = [], [], []
    t0 = time.perf_counter()
    n = 0

    for p_img, meta in iter_tiles(train_shards, a.tiles, a.seed):
        mask_np, _ = sample_mask(meta["tile"], rng, kind=a.mask_kind)
        if mask_np.sum() < 2000:
            continue
        img = torch.from_numpy(p_img.transpose(2, 0, 1)[None]).to(a.device)
        mask = torch.from_numpy(mask_np[None, None].astype(np.float32)).to(a.device)
        src = 1.0 - mask

        with torch.no_grad():
            q = make_query(sa, img, mask)
            cov = F.avg_pool2d(mask, BLOCK)                     # ブロックのマスク被覆率

            hard_idx = pick_hard_offsets(q, mask, src, offs_all, cov, r, N_HARD)
            remain = np.setdiff1d(np.arange(len(offs_all)), hard_idx)
            rand_idx = rng.choice(remain, size=N_RAND, replace=False)
            sel = np.concatenate([rand_idx, hard_idx]).astype(np.int64)

            # --- 真の誤差 (GT との差)。v1 と同じ定義。
            e, vfrac = [], []
            for i in sel:
                oy, ox = offs_all[int(i)]
                c = _shift(img, oy, ox)
                vs = _shift(src, oy, ox)
                # **有効性を誤差に混ぜない。** 誤差は「マスク内かつ参照元が有効」な
                # 画素だけで測り、有効割合は別に記録する (v1 と同じ理由)。
                use = mask * vs
                d2 = ((c - img) ** 2).mean(1, keepdim=True) * use
                num = F.avg_pool2d(d2, BLOCK)
                den = F.avg_pool2d(use, BLOCK)
                e.append(num / (den + 1e-8))
                vfrac.append(den / (cov + 1e-8))
            E = torch.cat(e, 1)[0].cpu().numpy()          # (N_SAMPLE, h, w)
            V = torch.cat(vfrac, 1)[0].cpu().numpy()
            COV = cov[0, 0].cpu().numpy()
            Q = q[0].permute(1, 2, 0).cpu().numpy()

        imgs.append((p_img * 4096).astype(np.int16))
        queries.append((Q * 4096).astype(np.int16))
        masks.append(np.packbits(mask_np))
        off_idx.append(sel.astype(np.int16))
        errs.append(E.astype(np.float32))
        covs.append(COV.astype(np.float32))
        vfracs.append(V.astype(np.float32))
        n += 1
        if n % 100 == 0:
            el = time.perf_counter() - t0
            p(f"  {n}/{a.tiles}  {el/n:.2f}s/タイル  残り約 {el/n*(a.tiles-n)/60:.0f}分")

    if not imgs:
        p("タイルが集まらなかった")
        return 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out, imgs=np.stack(imgs), queries=np.stack(queries), masks=np.stack(masks),
        off_idx=np.stack(off_idx), errs=np.stack(errs),
        covs=np.stack(covs), vfracs=np.stack(vfracs),
        offsets=np.array(offs_all, dtype=np.int16),
        block=BLOCK, radius=RADIUS, stride=STRIDE, img_scale=4096)
    p(f"\n{n} タイル -> {a.out} ({os.path.getsize(a.out)/1e6:.0f} MB)")

    E, C, V = np.stack(errs), np.stack(covs), np.stack(vfracs)
    blk = C > 0.25
    use = blk[:, None] & (V > 0.9)
    p(f"対象ブロック (マスク被覆 >25%): {int(blk.sum()):,} / {C.size:,}")
    p(f"  1ブロックあたり有効候補数 中央値 {np.median(use.sum(1)[blk]):.0f}/{N_SAMPLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
