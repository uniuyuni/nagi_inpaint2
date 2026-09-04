"""3 者盲検比較: LaMa / YOEO+L5 / SmartEraser (docs/DESIGN_V2_JA.md §6)。

**なぜ 3 者を並べるか** (ユーザ要望 2026-08-25)

2 者比較を 2 回やると判定の手間が倍になる。かといって
「自動指標で 1 つに絞ってから 2 者比較」は、**代理指標で選別してから人に見せる**
という v1 が 0勝16敗した構造そのものになる (§6.1)。
3 者を並べて「どれが一番良いか」を 1 つだけ選ぶ形なら、
判定の手間は 2 者比較とほぼ変わらず、代理指標も挟まない。

**表示** (ユーザ要望): 元画像 | マスク | A | B | C
マスクを並べるのは「何を消すはずだったか」を判定者が確認できるようにするため。

**「引き分け」を 2 つに分ける** (ユーザ指摘 2026-08-25)

`=` は「**どれも良い**」と「**全部ダメ**」の両方を含んでしまい、意味が正反対になる。

  - どれも良い → 3 手法とも解けている。判別できないだけで問題ではない
  - 全部ダメ   → 誰も解けていない。**残った仕事の所在はここ**

混ざったままだと伸びしろの位置が分からないので、記入を分ける:

    A / B / C  … 一番良いもの
    o          … どれも良い (差が無い)
    x          … 全部ダメ

**ケースの選び方と、その限界**

SmartEraser は 121 秒/箇所 (50 step + CFG) かかるので、全 60 件で回すと 2 時間になる。
そこで **LaMa と YOEO の出力差が大きい上位 N 件**を先に選び、
そこだけ SmartEraser を走らせる。

  - この基準は **LaMa と YOEO の間では対称** (どちらにも有利にならない)
  - **SmartEraser とは無関係** なので、有利にも不利にもならない
  - ただし「SmartEraser だけが大きく違うケース」は選ばれない。これは限界として記録する

  pixi run python eval/ab3.py --build --n 16
  # out/ab3/*.png を見て out/ab3/votes.txt に A/B/C/= を書く
  pixi run python eval/ab3.py --tally
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = ("lama", "yoeo", "smarteraser")
LABELS = {"lama": "LaMa", "yoeo": "YOEO+L5", "smarteraser": "SmartEraser"}


def p(*a):
    print(*a, flush=True)


def u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def build(bench: str, out: str, seed: int, device: str, n: int) -> None:
    from eval.lama_baseline import Lama
    from nagi_inpaint2.removers import SmartEraser, Yoeo, composite

    z = np.load(bench, allow_pickle=False)
    imgs, msks = z["img"], z["mask"]
    meta = json.loads(str(z["meta"]))
    N = len(meta)

    # --- 段1: 速い 2 者を全件に掛けて、差が大きいケースを選ぶ -----------------
    lama = Lama(device=device)
    yoeo = Yoeo(device=device, steps=2)
    fast = []
    t0 = time.perf_counter()
    for i in range(N):
        img = imgs[i].astype(np.float32)
        mask = msks[i].astype(bool)
        ol = composite(lama(img, mask), img, mask)
        oy = composite(yoeo(img, mask), img, mask)
        fast.append(dict(i=i, ol=ol, oy=oy,
                         spread=float(np.abs(ol[mask] - oy[mask]).mean())))
        if (i + 1) % 10 == 0:
            p(f"  段1 {i+1}/{N}  {(time.perf_counter()-t0)/(i+1):.1f} 秒/件")
    del yoeo
    import torch
    torch.mps.empty_cache()

    fast.sort(key=lambda c: -c["spread"])
    sel = fast[:n]
    p(f"\n段1 完了: LaMa と YOEO の差が大きい上位 {len(sel)} 件を選んだ")
    p(f"  差の範囲 {sel[-1]['spread']:.4f} 〜 {sel[0]['spread']:.4f}")

    # --- 段2: 選んだケースだけ SmartEraser を掛ける ---------------------------
    se = SmartEraser(device=device, steps=50, guidance=1.5)
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    key_map: dict[str, str] = {}
    t0 = time.perf_counter()

    for rank, c in enumerate(sel):
        i = c["i"]
        img = imgs[i].astype(np.float32)
        mask = msks[i].astype(bool)
        os_ = composite(se(img, mask), img, mask)

        outs = {"lama": c["ol"], "yoeo": c["oy"], "smarteraser": os_}
        order = list(METHODS)
        rng.shuffle(order)                       # A/B/C への割り当てを無作為化
        name = f"{rank:02d}_{os.path.basename(meta[i]['path'])}"
        key_map[name] = {"ABC"[k]: m for k, m in enumerate(order)}

        # 元画像 | マスク重ね | A | B | C
        ov = img.copy()
        ov[mask] = ov[mask] * 0.4 + np.array([1.0, 0.25, 0.25]) * 0.6
        panels = [img, ov] + [outs[m] for m in order]
        H, W = img.shape[:2]
        sheet = np.full((H, W * len(panels) + 4 * (len(panels) - 1), 3), 32, np.uint8)
        for j, pan in enumerate(panels):
            sheet[:, j * (W + 4):j * (W + 4) + W] = u8(pan)
        Image.fromarray(sheet).save(os.path.join(out, name + ".png"))
        el = time.perf_counter() - t0
        p(f"  段2 {rank+1}/{len(sel)}  {el/(rank+1):.0f} 秒/件  "
          f"残り約 {(len(sel)-rank-1)*el/(rank+1)/60:.0f} 分")

    json.dump({"cases": key_map, "seed": seed, "bench": bench},
              open(os.path.join(out, "key.json"), "w"))
    with open(os.path.join(out, "votes.txt"), "w") as f:
        f.write("# 各行: <ケース名> <A|B|C|o|x>\n"
                "# 列は 左から: 元画像 | マスク(赤) | A | B | C\n"
                "# **一番良いもの**を 1 つ選ぶ。観点:\n"
                "#   1 破綻がないか / 2 つながっているか / 3 消したいものが消えているか\n"
                "#   o = どれも良い (差が無い)\n"
                "#   x = 全部ダメ\n"
                "# ※ o と x は意味が正反対なので必ず分けること\n")
        for nm in key_map:
            f.write(f"{nm} \n")
    p(f"\n{len(sel)} 件 -> {out}/")
    p(f"  {out}/votes.txt に A / B / C / = を記入")
    p(f"  済んだら: pixi run python eval/ab3.py --tally")


def binom_p(k: int, n: int, p0: float) -> float:
    """片側二項検定 (k 以上になる確率)。3 者なら p0 = 1/3。"""
    return sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
               for i in range(k, n + 1))


def tally(out: str) -> None:
    key = json.load(open(os.path.join(out, "key.json")))["cases"]
    votes: dict[str, str] = {}
    for line in open(os.path.join(out, "votes.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        if len(t) > 1:
            votes[t[0]] = t[1].upper()

    win = {m: 0 for m in METHODS}
    allgood = allbad = unknown = 0
    for nm, v in votes.items():
        if nm not in key:
            continue
        if v in ("O", "○", "GOOD", "どれも良い"):
            allgood += 1
        elif v in ("X", "×", "BAD", "全部ダメ"):
            allbad += 1
        elif v in ("=", "同等", "TIE"):
            unknown += 1          # 旧記法。o/x に分けられていない
        elif v in key[nm]:
            win[key[nm][v]] += 1
    n = sum(win.values())
    tot = n + allgood + allbad + unknown
    p(f"投票 {len(votes)} 件 / 勝敗つき {n} 件")
    p(f"  どれも良い {allgood} 件 / 全部ダメ {allbad} 件"
      + (f" / 旧記法 = が {unknown} 件 (o か x に分けること)" if unknown else ""))
    if tot:
        p(f"  **全部ダメ = {allbad}/{tot} ({allbad/tot*100:.0f}%)** "
          f"<- 誰も解けていない割合。残った仕事の大きさ\n")
    p(f"{'手法':14s} {'1位':>5s} {'割合':>8s} {'偶然(33%)より上か':>20s}")
    for m in METHODS:
        r = win[m] / n if n else float("nan")
        pv = binom_p(win[m], n, 1 / 3) if n else float("nan")
        p(f"{LABELS[m]:14s} {win[m]:5d} {r*100:7.1f}% {pv:20.4f}")
    if n:
        best = max(METHODS, key=lambda m: win[m])
        p(f"\n  最多: {LABELS[best]} ({win[best]}/{n})  "
          f"片側二項検定 p = {binom_p(win[best], n, 1/3):.4f}")
        p(f"  ※ 3 者なので偶然の水準は 33.3%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--bench", default="cache/bench_removal.npz")
    ap.add_argument("--out", default="out/ab3")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    if a.build:
        build(a.bench, a.out, a.seed, a.device, a.n)
    elif a.tally:
        tally(a.out)
    else:
        ap.error("--build か --tally を指定すること")
    return 0


if __name__ == "__main__":
    sys.exit(main())
