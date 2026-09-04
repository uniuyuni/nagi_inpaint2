"""Day 3 / 擬似ペア候補のコンタクトシート (§9.2 の目視確認)。

data/pairs.py の出力は **候補生成器であって最終判定ではない**。
v1 では上位8組すべてが目視で不採用になった (露出変化・構図ずれ・別構図)。
必ずこのシートで確認してから採用すること。

ショットごとに最良1組を取るので、1ショットへの集中を避けて多様性を見られる。

  pixi run python data/pairs_sheet.py --n 8 --out out/pairs_sheet.png
  pixi run python data/pairs_sheet.py --n 8 --skip 8    # 次の8ショット
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw

PAIRS = "cache/pairs.jsonl"
HASHES = "cache/hash.jsonl"
WIDTH = 420


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--hashes", default=HASHES)
    ap.add_argument("--out", default="out/pairs_sheet.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--min-dt", type=float, default=0.0,
                    help="Δt の下限[秒]。連写を除きたいときに使う")
    a = ap.parse_args()

    jpeg = {}
    for line in open(a.hashes):
        try:
            r = json.loads(line)
            jpeg[r["path"]] = r.get("jpeg")
        except (json.JSONDecodeError, KeyError):
            continue

    pairs = [json.loads(l) for l in open(a.pairs)]
    pairs = [p for p in pairs if p["dt_sec"] >= a.min_dt]

    best = {}
    for p in pairs:                       # 既に良い順に並んでいる
        best.setdefault(p["shoot"], p)
    sel = list(best.values())[a.skip:a.skip + a.n]
    if not sel:
        print("候補なし")
        return 1

    rows = []
    for p in sel:
        ims = []
        for k in ("a", "b"):
            path = jpeg.get(p[k])
            if not path or not os.path.exists(path):
                break
            with Image.open(path) as im:
                im.draft("RGB", (im.width // 8, im.height // 8))
                im = im.convert("RGB")
                im.thumbnail((WIDTH, WIDTH))
                ims.append(im)
        if len(ims) == 2:
            rows.append((ims, p))

    if not rows:
        print("画像を開けなかった")
        return 1

    w = WIDTH * 2 + 8
    h = sum(max(i.height for i in ims) + 26 for ims, _ in rows)
    sheet = Image.new("RGB", (w, h), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    y = 0
    for ims, p in rows:
        x = 0
        for im in ims:
            sheet.paste(im, (x, y))
            x += WIDTH + 8
        y += max(i.height for i in ims)
        d.text((4, y + 6),
               f"{p['shoot']}  compact {p['compact']:.2f} diff {p['diff_cells']:.3f} "
               f"shift {p['shift_gain']:+.4f} dt {p['dt_sec']:.0f}s  "
               f"{os.path.basename(p['a'])} <-> {os.path.basename(p['b'])}",
               fill=(230, 230, 230))
        y += 26

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sheet.save(a.out)
    print(f"{a.out}  {sheet.size}  ({len(rows)} 組)")
    for _, p in rows:
        print(f"  {p['shoot']}: {os.path.basename(p['a'])} <-> {os.path.basename(p['b'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
