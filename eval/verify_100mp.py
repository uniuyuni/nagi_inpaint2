"""1 億画素での通し確認 (SPEC §1.1 / §9.1)。

**確かめること**

1. **マスク外が原画素とビット一致**すること (両方の候補で)
2. **費用が画像の画素数に依存しない**こと — 512² と 100MP で同じ時間になるか
3. ピークメモリが M1 16GB に収まること

`patches()` は変更した矩形だけを返すので、画像全体の複製が起きない。
`__call__` は全体を候補の数だけ複製するので **100MP では使ってはいけない**
(1.2GB × 候補数)。この差もここで測る。

  pixi run python eval/verify_100mp.py
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def p(*a):
    print(*a, flush=True)


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def make(side: int, seed: int = 0):
    """知覚空間の合成画像とマスク。**中身は本物でなくてよい** — ここで測るのは
    費用と不変条件であって品質ではない (品質は `eval/accept.py` で測った)。"""
    rng = np.random.default_rng(seed)
    # 一様乱数だと自動露出が飽和するので、緩やかな諧調 + 粒
    y = np.linspace(0.15, 0.75, side, dtype=np.float32)[:, None]
    x = np.linspace(-0.1, 0.1, side, dtype=np.float32)[None, :]
    base = (y + x)[..., None] * np.array([1.0, 0.97, 0.92], np.float32)
    img = np.ascontiguousarray(
        base + rng.normal(0, 0.02, (side, side, 3)).astype(np.float32))
    mask = np.zeros((side, side), bool)
    cy, cx = side // 2, side // 2
    mask[cy - 90:cy + 90, cx - 40:cx + 40] = True
    return img, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--big", type=int, default=10240, help="大きい方の一辺")
    a = ap.parse_args()

    from nagi_inpaint2.pipeline import MultiInpainter

    p(f"起動時 RSS {rss():.2f}GB")
    t0 = time.perf_counter()
    multi = MultiInpainter(device=a.device)
    p(f"モデル読み込み {time.perf_counter()-t0:.1f} 秒 / RSS {rss():.2f}GB "
      f"({' + '.join(multi.cores)})\n")

    rows = []
    for side in (512, 2048, a.big):
        mp = side * side / 1e6
        img, mask = make(side)
        p(f"--- {side}² = {mp:.1f}MP  (配列 {img.nbytes/1e9:.2f}GB)")
        t0 = time.perf_counter()
        y0, x0, pats = multi.patches(img, mask)
        dt = time.perf_counter() - t0

        # (1) マスク外がビット一致か
        exact = {}
        sub = img[y0:y0 + pats[multi.cores[0]].shape[0],
                  x0:x0 + pats[multi.cores[0]].shape[1]]
        sm = mask[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
        for c, q in pats.items():
            exact[c] = bool((q[~sm] == sub[~sm]).all())
        # 穴の中は必ず変わっていること (何もしていないのを見逃さないため)
        changed = {c: float(np.abs(q[sm] - sub[sm]).mean())
                   for c, q in pats.items()}
        rows.append((mp, dt, rss()))
        p(f"  patches() {dt:.2f} 秒 / RSS {rss():.2f}GB / "
          f"パッチ {sub.shape[0]}²")
        for c in multi.cores:
            p(f"    {c:6s} マスク外ビット一致 {'OK' if exact[c] else '**NG**'}"
              f"   穴の中の変化量 {changed[c]:.4f}")
        if not all(exact.values()):
            p("  **マスク外が書き換わっている。SPEC §9.1 違反**")
            return 1
        if min(changed.values()) < 1e-6:
            p("  **穴が埋まっていない**")
            return 1
        del img, mask, pats

    p("\n=== 費用が画素数に依存しないこと ===")
    p(f"{'画素数':>10s} {'patches()':>12s} {'基準比':>8s}")
    base = rows[0][1]
    for mp, dt, _ in rows:
        p(f"{mp:9.1f}MP {dt:11.2f}s {dt/base:7.2f}x")

    # __call__ との差 (なぜ patches を使うのか)
    img, mask = make(2048)
    t0 = time.perf_counter(); multi.patches(img, mask); t1 = time.perf_counter()
    res = multi(img, mask); t2 = time.perf_counter()
    p(f"\n=== なぜ patches() を使うのか (4.2MP で比較) ===")
    p(f"  patches()  {t1-t0:.2f} 秒 (返すのは変更した矩形だけ)")
    p(f"  __call__() {t2-t1:.2f} 秒 (画像全体を候補数だけ複製: "
      f"{sum(v.nbytes for v in res.values())/1e9:.2f}GB)")
    p(f"  100MP・2案なら複製だけで {10240**2*3*4*2/1e9:.1f}GB になる")
    p(f"\n最終 RSS {rss():.2f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
