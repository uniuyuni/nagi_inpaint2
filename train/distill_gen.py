"""Phase 0b — 蒸留の教師データを事前生成する (docs/DESIGN_V2_JA.md §4.2.1)。

**この実験が答える問い**

v2 設計の中核主張は次のとおり:

> 目標が「正解画像」だと、穴の中身は入力から復元不能なので条件付き分布が広がり、
> L1 の最適解は平均＝ぼけになる。目標を「決定的な教師の出力」に替えれば
> 条件付き分布はデルタ関数になり、**L1 で学習してもぼけない**。

この主張は目標側については厳密だが、**生徒側に 2 つの抜け穴がある** (§4.2.1):

  (1) 生徒の容量が有限で教師 T が入力に対して非滑らかなら、
      生徒は近傍の X に対する T の平均を出す = 入力空間での平均化で再びぼける
  (2) 有限ステップの SGD がその最適解に到達する保証はない (スペクトルバイアス)。
      **v1 の実測が逆方向の証拠**: Stage G は grain が step 4500 で頭打ちになり、
      9,000 step まで回しても「粒は粒のまま、構造にならなかった」

**LaMa は既に手元にあり (`models/big-lama.pt`)、完全に決定的な教師である。**
拡散モデルを一切ダウンロードせずに、上の 2 点を実測で潰せる。
教師データ生成 4〜7 日を始める前に、1 日で中核主張を反証できる (SPEC R13/R14)。

**このスクリプトが作るもの**

同一のタイル・同一のマスクに対して以下を 1 つの npz に詰める。
2 つの arm (目標=正解画像 / 目標=LaMa出力) が**完全に同じ入力を見る**ことが
対照実験の条件なので、両者を 1 ファイルに入れて取り違えを不可能にする。

    img    (N,3,C,C) float16  入力タイル (知覚空間)
    lama   (N,3,C,C) float16  LaMa の出力 (マスク内外を問わず生出力)
    mask   (N,1,C,C) uint8    マスク (0/1)
    meta                      shard / key / mask_seed / kind / bbox

**設計判断**

- **LaMa は 512 の全タイルに掛け、保存は 256 クロップにする。**
  教師には本来の文脈を与えつつ、生徒の学習は 256 で回して時間を節約する。
  クロップはマスク外接矩形の中心にジッタを付けて取る (運用でも
  「マスク周りを切り出す」ので分布が一致する)。
- **マスク外接矩形が `--max-bbox` を超えるものは捨てる。**
  256 クロップ内に `grain_ratio` の annulus (32px) と内側侵食 (3px) が
  収まらないと指標が nan を返すため (SPEC R12 で見つけた grain の盲点)。
- **ベンチに使う保留 shard は除外する。** 評価セットへの学習データ混入は
  SPEC §12.2 で 1 度やっている。
- **shard は無作為に選び、1 shard あたりの採取数を制限する。**
  `train/dataset.py:49` の警告どおり、shard は (機種, ショット) 順に
  詰まっているので先頭から順に読むと **train が X-E3 だけ・val が X-T5 だけ**
  になる (初版で実際にそうなった)。学習と保留を同じ分布から取るため、
  1 回の実行で train/val を **shard 単位で分割**して両方作る。

  pixi run python train/distill_gen.py --n 4000 --out cache/distill_train.npz
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import tarfile
import time

import warnings

import numpy as np
import torch
import zstandard as zstd

warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.lama_baseline import Lama                              # noqa: E402
from masks import sample_mask                                    # noqa: E402
from nagi_inpaint2.color import byte_unshuffle, dequantize       # noqa: E402


def p(*a):
    print(*a, flush=True)


def bench_shards() -> set[str]:
    """ベンチ定義が使っている保留 shard を集める (学習に使わないため)。"""
    out: set[str] = set()
    for f in glob.glob("cache/bench_*.json") + glob.glob("cache/benchmark*.json"):
        try:
            spec = json.load(open(f))
        except Exception:
            continue
        out.update(spec.get("shards") or [])
        out.update(c["shard"] for c in spec.get("cases", []))
    return out


def iter_tiles(path: str, dctx):
    """1 shard を順に舐めて (key, img, meta) を返す。tar は逐次読みが最速。"""
    with tarfile.open(path) as tf:
        group: dict[str, dict] = {}
        for member in tf:
            if not member.isfile():
                continue
            key, _, ext = member.name.partition(".")
            g = group.setdefault(key, {})
            g[ext] = tf.extractfile(member).read()
            if len(g) < 3:
                continue
            group.pop(key)
            try:
                meta = json.loads(g["json"])
                img = byte_unshuffle(dctx.decompress(g["img.u16.zst"]),
                                     tuple(meta["shape"]))
                img = dequantize(img, meta["q_scale"], meta["q_offset"])
            except Exception:
                continue
            yield key, img, meta


def crop_around(mask: np.ndarray, size: int, rng: np.random.Generator):
    """マスク外接矩形の中心にジッタを付けた `size` 角のクロップ範囲を返す。"""
    ys, xs = np.nonzero(mask)
    cy = int((ys.min() + ys.max()) // 2)
    cx = int((xs.min() + xs.max()) // 2)
    h, w = mask.shape
    # 外接矩形が収まる範囲でジッタ (最大 ±size/8)
    j = size // 8
    cy += int(rng.integers(-j, j + 1))
    cx += int(rng.integers(-j, j + 1))
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return y0, x0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--n", type=int, default=4000, help="生成する組数")
    ap.add_argument("--crop", type=int, default=256, help="保存するクロップ辺")
    ap.add_argument("--max-bbox", type=int, default=176,
                    help="マスク外接矩形の上限。grain の annulus を確保するため")
    ap.add_argument("--min-area", type=int, default=400, help="マスク画素数の下限")
    ap.add_argument("--kinds", default="blob,rect,silhouette",
                    help="使うマスク種別。主用途は塊状なので既定は塊系のみ。"
                         "細い種別 (wire/stroke/dust) は外接矩形が大きく "
                         "--max-bbox で全て落ちるため、初めから含めない")
    ap.add_argument("--out", default="cache/distill_train.npz")
    ap.add_argument("--val-out", default="cache/distill_val.npz")
    ap.add_argument("--val-n", type=int, default=400, help="保留の件数")
    ap.add_argument("--per-shard", type=int, default=40,
                    help="1 shard から採る上限。機種・ショットの偏りを防ぐ")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=0, help="0 なら制限なし")
    a = ap.parse_args()

    held = bench_shards()
    allsh = sorted(f for f in os.listdir(a.shards) if f.endswith(".tar"))
    use = [s for s in allsh if s not in held]
    p(f"shard {len(allsh)} 本中 {len(held)} 本がベンチ用 -> 学習に {len(use)} 本")

    # **無作為に並べ替える。先頭から順に読むと機種が偏る** (train/dataset.py:49)
    import random as _random
    _random.Random(a.seed).shuffle(use)
    if a.max_shards:
        use = use[:a.max_shards]

    # train/val を shard 単位で分ける (同一画像由来のタイルが両側に入らない)
    n_val_sh = max(1, int(round(len(use) * 0.12)))
    val_sh, tr_sh = use[:n_val_sh], use[n_val_sh:]
    p(f"無作為化して train {len(tr_sh)} 本 / val {len(val_sh)} 本に分割 "
      f"(1 shard あたり最大 {a.per_shard} 件)")

    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    p(f"マスク種別: {kinds} (主用途の塊状に絞る。細い種別は §4.4 の別課題)")

    lama = Lama(device=a.device)
    dctx = zstd.ZstdDecompressor()

    C = a.crop

    def collect(shard_list, want: int, seed: int, tag: str):
        rng = np.random.default_rng(seed)
        imgs = np.zeros((want, 3, C, C), np.float16)
        lams = np.zeros((want, 3, C, C), np.float16)
        msks = np.zeros((want, 1, C, C), np.uint8)
        rows: list[dict] = []
        n = n_bbox = n_area = 0
        t0 = time.perf_counter()
        for shard in shard_list:
            if n >= want:
                break
            got = 0
            for key, img, meta in iter_tiles(os.path.join(a.shards, shard), dctx):
                if n >= want or got >= a.per_shard:
                    break
                tile = meta["tile"]
                if tile < C:
                    continue
                mask_seed = int(rng.integers(0, 2 ** 31 - 1))
                kind = str(rng.choice(kinds))
                mask, mmeta = sample_mask(tile, np.random.default_rng(mask_seed),
                                          kind=kind)
                if mask.sum() < a.min_area:
                    n_area += 1
                    continue
                ys, xs = np.nonzero(mask)
                bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
                if max(bh, bw) > a.max_bbox:
                    n_bbox += 1
                    continue

                y = lama(img, mask)          # 教師には 512 の全タイルを見せる
                y0, x0 = crop_around(mask, C, rng)
                sl = (slice(y0, y0 + C), slice(x0, x0 + C))
                cm = mask[sl]
                if cm.sum() < a.min_area:
                    continue
                imgs[n] = img[sl].transpose(2, 0, 1).astype(np.float16)
                lams[n] = y[sl].transpose(2, 0, 1).astype(np.float16)
                msks[n] = cm[None].astype(np.uint8)
                rows.append(dict(shard=shard, key=key, mask_seed=mask_seed,
                                 kind=mmeta["kind"], bh=int(bh), bw=int(bw),
                                 y0=y0, x0=x0))
                n += 1
                got += 1
                if n % 200 == 0:
                    el = time.perf_counter() - t0
                    p(f"  [{tag}] {n}/{want}  {el/n:.2f} s/件  "
                      f"残り約 {(want-n)*el/n/60:.0f} 分")
        p(f"  [{tag}] {n} 件 / 除外 面積 {n_area} 外接矩形 {n_bbox}")
        return imgs[:n], lams[:n], msks[:n], rows

    def save(path, imgs, lams, msks, rows, tag):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, img=imgs, lama=lams, mask=msks,
                 meta=json.dumps(rows), crop=C, seed=a.seed,
                 kinds=",".join(kinds))
        sz = os.path.getsize(path) / 1e9
        bodies: dict[str, int] = {}
        shoots = set()
        for r in rows:
            b = r["key"].split("_")[0]
            bodies[b] = bodies.get(b, 0) + 1
            shoots.add("_".join(r["key"].split("_")[:2]))
        p(f"{tag}: {len(rows)} 件 -> {path} ({sz:.2f} GB)")
        p(f"  機種 {bodies}  元画像 {len(shoots)} 枚")

    ti, tl, tm, trows = collect(tr_sh, a.n, a.seed, "train")
    save(a.out, ti, tl, tm, trows, "train")
    del ti, tl, tm
    vi, vl, vm, vrows = collect(val_sh, a.val_n, a.seed + 999, "val")
    save(a.val_out, vi, vl, vm, vrows, "val")

    kinds_t: dict[str, int] = {}
    for r in trows:
        kinds_t[r["kind"]] = kinds_t.get(r["kind"], 0) + 1
    p(f"\nマスク種別 (train): {kinds_t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
