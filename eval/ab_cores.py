"""除去コアどうしの盲検A/B (docs/DESIGN_V2_JA.md §6)。

`eval/ab_removal.py` は LaMa 対 YOEO に固定で書かれていた。
**コアの入れ替えを試す段**に入ったので、任意の 2 つを比べられる形にする
(`--a yoeo --b rorem` など)。判定の作法は `ab_removal` / `ab_context` と同じ:

  - 表示は **元画像 | マスク(赤) | A | B**。4 コマとも同じ場所・同じ画角
  - 絞り込みは**両者の出力差が大きい順**。完全に対称な基準 (R20)
  - A/B の割り当ては**厳密に半々** (コイン投げは 20 件で 5/15 の偏りが出る、実測)
  - `o` = どちらも良い / `x` = どちらもダメ。**この 2 つは意味が正反対**なので分ける

L5 (Poisson 合成) は**両者に掛ける**。LaMa には実質無効なので公平である。

  pixi run python eval/ab_cores.py --build --a yoeo --b rorem --n 20
  pixi run python eval/ab_cores.py --tally
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


def p(*a):
    print(*a, flush=True)


def u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def _kw(name: str) -> dict:
    """コアごとの既定の推論設定 (実測で決めたもの)。"""
    if name == "yoeo":
        return dict(steps=2)              # 3/4 step は実測で改善せず
    return {}


def run_core(name: str, bench: str, device: str, cache: str) -> np.ndarray:
    """1 つのコアで全件を埋め、L5 まで通した結果を返す。**1 コアずつ**回す。

    2 つを同時に載せると SDXL (fp16 で 6.9GB) が M1 16GB に収まらないので、
    片方ずつ走らせて中間結果を `cache` に置く。
    """
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        p(f"  {name}: {cache} から読み込み ({len(z['out'])} 件)")
        return z["out"]

    from nagi_inpaint2.removers import build, composite
    z = np.load(bench, allow_pickle=False)
    imgs, msks = z["img"], z["mask"]
    core = build(name, device=device, **_kw(name))
    out = np.zeros(imgs.shape, np.float16)
    t0 = time.perf_counter()
    for i in range(len(imgs)):
        img = imgs[i].astype(np.float32)
        m = msks[i].astype(bool)
        o = composite(core(img, m), img, m, poisson=True)
        o[~m] = img[~m]                   # マスク外はビット単位で原画素
        out[i] = o.astype(np.float16)
        if (i + 1) % 5 == 0:
            el = time.perf_counter() - t0
            p(f"  {name} {i+1}/{len(imgs)}  {el/(i+1):.1f} 秒/件  "
              f"残り約 {(len(imgs)-i-1)*el/(i+1)/60:.0f} 分")
    np.savez_compressed(cache, out=out)
    p(f"  {name}: {time.perf_counter()-t0:.0f} 秒 -> {cache}")
    del core
    import torch
    if device == "mps":
        torch.mps.empty_cache()
    return out


def build_sheets(bench: str, out: str, a: str, b: str, seed: int, device: str,
                 n: int, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(bench))[0]
    OA = run_core(a, bench, device, os.path.join(cache_dir, f"{tag}_{a}.npz"))
    OB = run_core(b, bench, device, os.path.join(cache_dir, f"{tag}_{b}.npz"))

    z = np.load(bench, allow_pickle=False)
    imgs, msks = z["img"], z["mask"]
    meta = json.loads(str(z["meta"]))

    cases = []
    for i in range(len(meta)):
        m = msks[i].astype(bool)
        oa, ob = OA[i].astype(np.float32), OB[i].astype(np.float32)
        cases.append(dict(i=i, name=f"{i:02d}_{os.path.basename(meta[i]['path'])}",
                          img=imgs[i].astype(np.float32), mask=m, oa=oa, ob=ob,
                          diff=float(np.abs(oa[m] - ob[m]).mean())))
    cases.sort(key=lambda c: -c["diff"])
    if 0 < n < len(cases):
        cases = cases[:n]
        p(f"\n出力差が大きい上位 {len(cases)} 件に絞った (両者に対称な基準)")
        p(f"  差の範囲 {cases[-1]['diff']:.4f} 〜 {cases[0]['diff']:.4f}")

    # --- A/B を厳密に半々に割り当てる (R20) ---
    rng = random.Random(seed)
    flags = [True] * (len(cases) // 2) + [False] * (len(cases) - len(cases) // 2)
    rng.shuffle(flags)

    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".png"):
            os.remove(os.path.join(out, f))

    key_map: dict[str, str] = {}
    for rank, (c, a_is_A) in enumerate(zip(cases, flags)):
        P, Q = (c["oa"], c["ob"]) if a_is_A else (c["ob"], c["oa"])
        key_map[c["name"]] = "A" if a_is_A else "B"
        img, m = c["img"], c["mask"]
        ov = img.copy()
        ov[m] = ov[m] * 0.4 + np.array([1.0, .25, .25]) * 0.6
        H, W = img.shape[:2]
        sheet = np.full((H, W * 4 + 12, 3), 32, np.uint8)
        for j, pan in enumerate((img, ov, P, Q)):
            sheet[:, j * (W + 4):j * (W + 4) + W] = u8(pan)
        Image.fromarray(sheet).save(
            os.path.join(out, f"{rank:02d}_{c['name']}.png"))

    json.dump({"cases": key_map, "seed": seed, "bench": bench, "a": a, "b": b},
              open(os.path.join(out, "key.json"), "w"))
    with open(os.path.join(out, "votes.txt"), "w") as f:
        f.write("# 各行: <ケース名> <A|B|o|x>\n"
                "# 列は 左から: 元画像 | マスク(赤) | A | B\n"
                "# 観点: 破綻がないか / つながっているか / 消えているか\n"
                "#   o = どちらも良い / x = どちらもダメ\n")
        for nm in key_map:
            f.write(f"{nm} \n")
    na = sum(1 for v in key_map.values() if v == "A")
    p(f"\n{len(key_map)} 件 -> {out}/   ({a} が A: {na}/{len(key_map)})")
    p(f"  済んだら: pixi run python eval/ab_cores.py --tally --out {out}")


def tally(out: str) -> None:
    k = json.load(open(os.path.join(out, "key.json")))
    key, a, b = k["cases"], k.get("a", "A側"), k.get("b", "B側")
    votes: dict[str, str] = {}
    for line in open(os.path.join(out, "votes.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        if len(t) > 1:
            votes[t[0]] = t[1].upper()

    wa = wb = good = bad = 0
    for nm, v in votes.items():
        if nm not in key:
            continue
        if v in ("O", "○", "0"):
            good += 1
        elif v in ("X", "×"):
            bad += 1
        elif v in ("A", "B"):
            wa, wb = (wa + 1, wb) if v == key[nm] else (wa, wb + 1)
    n = wa + wb
    tot = n + good + bad
    p(f"投票 {len(votes)} 件 / 勝敗つき {n} 件")
    p(f"  どちらも良い {good} 件 / どちらもダメ {bad} 件"
      + (f"  (どちらもダメ = {bad}/{tot} = {bad/tot*100:.0f}%)" if tot else ""))
    if not n:
        p("\n判定なし")
        return
    pv = min(1.0, sum(math.comb(n, i) * 0.5 ** n
                      for i in range(min(wa, wb) + 1)) * 2)
    p(f"\n=== {a} 対 {b} ===")
    p(f"  {a} {wa} 勝 / {b} {wb} 勝   {a} の勝率 {wa/n*100:.1f}% (n={n})")
    p(f"  両側二項検定 p = {pv:.4f}")
    win, lose = (a, b) if wa > wb else (b, a)
    if wa == wb:
        p("  -> 互角")
    elif pv < 0.05:
        p(f"  -> **{win} が有意に良い**")
    else:
        p(f"  -> {win} が優勢だが有意ではない (件数不足)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--a", default="yoeo")
    ap.add_argument("--b", default="rorem")
    ap.add_argument("--bench", default="cache/bench_removal.npz")
    ap.add_argument("--out", default="out/ab_cores")
    ap.add_argument("--cache-dir", default="cache/core_out")
    ap.add_argument("--n", type=int, default=20, help="0 なら全件")
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--device", default="mps")
    x = ap.parse_args()

    if x.build:
        build_sheets(x.bench, x.out, x.a, x.b, x.seed, x.device, x.n,
                     x.cache_dir)
    elif x.tally:
        tally(x.out)
    else:
        ap.error("--build か --tally を指定すること")
    return 0


if __name__ == "__main__":
    sys.exit(main())
