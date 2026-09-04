"""Day 3 / 知覚ハッシュ — 隣接カメラ JPEG から近重複を検出する (SPEC §6.4, §9.2)。

**JPEG を使ってよい理由**: 知覚ハッシュはトーンマップの影響を受けない。
品質判定 (白飛び・シャープネス・ノイズ) に JPEG を使ってはならないが
(理由は data/score_raw.py の docstring)、同一性・類似性の判定には妥当で、
RAW を開くより桁違いに速い。

用途は2つ:

  1. **重複排除** — 連写やブラケットで実質同じ画像を複数選抜しないようにする
  2. **擬似ペア発掘 (§9.2)** — 評価用のペア撮影データが存在しないため、
     「同一シーンで被写体だけが動いた前後」を既存ライブラリから掘り出す。
     同一フォルダ = 同一ショットなので、探索はフォルダ内に閉じる
     (全画像の総当たりに対し探索空間が数万分の1になる)

出力: cache/hash.jsonl。中断・再開可能。グループ化は data/pairs.py が行う。

  pixi run python data/score_jpeg.py
  pixi run python data/score_jpeg.py --limit 200

注意: /Volumes/PHOTOS 01 は読み出しのみ。一切書き込まない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

INDEX = "cache/index.jsonl"
OUT = "cache/hash.jsonl"

DRAFT = 8       # JPEG の DCT 段階で 1/8 に間引いてデコードする (高速)
HASH_W = 9      # dHash: 9x8 に縮小して横隣接比較 -> 64bit
HASH_H = 8
TINY = 16       # 粗い輝度サムネイル。ハッシュより緩い類似度に使う


def p(*a):
    print(*a, flush=True)


def hashes(path: str) -> dict:
    with Image.open(path) as im:
        w, h = im.size
        # draft() は JPEG デコード時に DCT 係数段階で縮小するため、
        # フルデコードしてから resize するより一桁速い
        im.draft("L", (max(w // DRAFT, 1), max(h // DRAFT, 1)))
        g = im.convert("L")

        d = np.asarray(g.resize((HASH_W, HASH_H), Image.BILINEAR), dtype=np.int16)
        dh = (d[:, 1:] > d[:, :-1]).flatten()
        dhash = int("".join("1" if b else "0" for b in dh), 2)

        a = np.asarray(g.resize((HASH_H, HASH_H), Image.BILINEAR), dtype=np.float32)
        ah = (a > a.mean()).flatten()
        ahash = int("".join("1" if b else "0" for b in ah), 2)

        t = np.asarray(g.resize((TINY, TINY), Image.BILINEAR), dtype=np.float32)
        t = (t - t.mean()) / (t.std() + 1e-6)

        return dict(dhash=dhash, ahash=ahash,
                    tiny=[round(float(v), 3) for v in t.flatten()],
                    jpeg_size=[w, h])


def load_jsonl(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def load_done(out: str) -> set[str]:
    if not os.path.exists(out):
        return set()
    done = set()
    with open(out) as f:
        for line in f:
            try:
                done.add(json.loads(line)["path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(a.index):
        p(f"索引がない: {a.index}  先に data/index.py を実行すること")
        return 1

    recs = [r for r in load_jsonl(a.index) if r.get("jpeg")]
    p(f"隣接 JPEG のある画像 {len(recs):,} 件")
    if a.limit:
        recs = recs[:a.limit]

    done = load_done(a.out)
    todo = [r for r in recs if r["path"] not in done]
    if done:
        p(f"  処理済み {len(done):,} 件をスキップ -> 残り {len(todo):,} 件")
    if not todo:
        p("完了済み")
        return 0

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.perf_counter()
    n_err = 0
    with open(a.out, "a") as fo:
        for i, r in enumerate(todo, 1):
            try:
                rec = hashes(r["jpeg"])
            except Exception as e:
                n_err += 1
                rec = dict(error=f"{type(e).__name__}: {e}")
            rec.update(path=r["path"], jpeg=r["jpeg"], body=r.get("body"),
                       shoot=r.get("shoot"), datetime=r.get("datetime"))
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                fo.flush()
                el = time.perf_counter() - t0
                p(f"  {i:>6,}/{len(todo):,}  {el/60:5.1f}分  {el/i*1000:.0f}ms/枚  "
                  f"残り約 {el/i*(len(todo)-i)/60:5.1f}分  エラー {n_err}")

    p(f"\n完了: {a.out}  {(time.perf_counter()-t0)/60:.1f}分  エラー {n_err} 件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
