"""**採用できるか**を 1 件ずつ聞く (docs/DESIGN_V2_JA.md §6、R23)。

**なぜ A/B ではないのか**

ShadeWave は結果を見せて採否をユーザに委ねられるので、L6 検品を作らないことにした
(2026-08-30)。その結果、**測るべき数字が「LaMa への勝率」から「採用率」に変わった**。

これまでの判定は全て比較 (A/B・3 者) だった。比較は
「**どちらが良いか**」しか聞いていないので、**負けた方が「不採用」なのか
「劣るが採用可」なのかが記録されていない**。記録漏れではなく、その質問を一度もしていない。

そのせいで次が決まらない:

  - LaMa 単独の採用率が 7 割あるなら -> 0.63 秒で足り、拡散モデルは要らない
  - LaMa が 4 割で LaMa+YOEO が 7 割なら -> YOEO は 5.3 秒を払う価値がある

**したがって「どちらが良いか」ではなく「それぞれ採用できるか」を独立に聞く。**

**記入形式**: 1 行に 2 文字。`[Aの可否][Bの可否]`、`o` = 採用できる / `x` = できない。

    oo  両方採用できる      ox  左(A)だけ
    xo  右(B)だけ          xx  両方ダメ

**左右は無作為化して伏せる。** 絶対評価でも「凝っている方を選ぶ」偏りは出るため。
割り当ては**厳密に半々** (R20。コイン投げは 20 件で 5/15 の偏りが出た実績がある)。

**全 60 件を使う。** これまでの判定は全て「差が大きい上位 N 件」という
偏った集合の上での値だった (採用率の見積もりが 62.5% 〜 85% とばらついたのはこのため)。
**偏っていない集合での率を初めて測る。**

出力はすべて `cache/core_out/` にキャッシュ済みなので、**新しい計算は発生しない**。

  pixi run python eval/accept.py --build
  # out/accept/*.png を見て out/accept/votes.txt に oo/ox/xo/xx を書く
  pixi run python eval/accept.py --tally
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, NG = "o", "x"
ALIAS = {"o": OK, "○": OK, "0": OK, "0".translate(str.maketrans("0", "0")): OK,
         "x": NG, "×": NG, "X": NG, "ｘ": NG}


def p(*a):
    print(*a, flush=True)


def u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def build(bench: str, out: str, a: str, b: str, seed: int,
          cache_dir: str) -> None:
    z = np.load(bench, allow_pickle=False)
    imgs, msks = z["img"], z["mask"]
    meta = json.loads(str(z["meta"]))
    tag = os.path.splitext(os.path.basename(bench))[0]

    O = {}
    for name in (a, b):
        f = os.path.join(cache_dir, f"{tag}_{name}.npz")
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"{f} が無い。先に出力を作ること:\n"
                f"  pixi run python eval/ab_cores.py --build --a {a} --b {b} --n 0")
        O[name] = np.load(f, allow_pickle=False)["out"]

    n = len(meta)
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)                      # 並び順にも意味を持たせない
    # **厳密に半々** (R20)
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flags)

    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".png"):
            os.remove(os.path.join(out, f))

    key: dict[str, str] = {}
    for rank, (i, a_is_A) in enumerate(zip(order, flags)):
        img = imgs[i].astype(np.float32)
        m = msks[i].astype(bool)
        P = O[a][i].astype(np.float32)
        Q = O[b][i].astype(np.float32)
        L, R = (P, Q) if a_is_A else (Q, P)
        nm = f"{i:02d}_{os.path.basename(meta[i]['path'])}"
        key[nm] = a if a_is_A else b          # A 側がどちらか

        ov = img.copy()
        ov[m] = ov[m] * 0.4 + np.array([1.0, .25, .25]) * 0.6
        H, W = img.shape[:2]
        sheet = np.full((H, W * 4 + 12, 3), 32, np.uint8)
        for j, pan in enumerate((img, ov, L, R)):
            sheet[:, j * (W + 4):j * (W + 4) + W] = u8(pan)
        Image.fromarray(sheet).save(os.path.join(out, f"{rank:02d}_{nm}.png"))

    json.dump({"cases": key, "seed": seed, "bench": bench, "a": a, "b": b},
              open(os.path.join(out, "key.json"), "w"))
    with open(os.path.join(out, "votes.txt"), "w") as f:
        f.write("# 各行: <ケース名> <2文字>\n"
                "# 列は 左から: 元画像 | マスク(赤) | A | B\n"
                "# 1文字目 = A の可否、2文字目 = B の可否\n"
                "#   o = 採用できる (この結果をそのまま使える)\n"
                "#   x = 採用できない (破綻・つながっていない・消えていない)\n"
                "#   oo=両方  ox=Aだけ  xo=Bだけ  xx=両方ダメ\n"
                "# ※ **比較ではありません。** 片方が明らかに良くても、\n"
                "#    もう片方も使えるなら oo と書いてください\n")
        for nm in key:
            f.write(f"{nm} \n")
    na = sum(1 for v in key.values() if v == a)
    p(f"{len(key)} 件 -> {out}/   ({a} が A 側: {na}/{len(key)})")
    p(f"  新しい計算は発生していない (出力は {cache_dir} のキャッシュ)")
    p(f"  済んだら: pixi run python eval/accept.py --tally")


def tally(out: str) -> None:
    k = json.load(open(os.path.join(out, "key.json")))
    key, a, b = k["cases"], k["a"], k["b"]
    got: dict[str, dict[str, str]] = {}
    bad_lines = []
    for line in open(os.path.join(out, "votes.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        if len(t) < 2:
            continue
        nm, v = t[0], t[1]
        ch = [ALIAS.get(c) for c in v]
        if nm not in key or len(ch) != 2 or None in ch:
            bad_lines.append(line.rstrip())
            continue
        side_a = key[nm]                      # A 側の手法名
        side_b = b if side_a == a else a
        got[nm] = {side_a: ch[0], side_b: ch[1]}

    if bad_lines:
        p(f"※ 読めなかった行 {len(bad_lines)} 件 (2文字の o/x が必要):")
        for l in bad_lines[:5]:
            p(f"    {l}")
    n = len(got)
    if not n:
        p("判定なし")
        return

    ka = sum(1 for d in got.values() if d[a] == OK)
    kb = sum(1 for d in got.values() if d[b] == OK)
    both = sum(1 for d in got.values() if d[a] == OK or d[b] == OK)
    n_ab = sum(1 for d in got.values() if d[a] == OK and d[b] == OK)
    only_a = ka - n_ab
    only_b = kb - n_ab
    none = n - both

    p(f"判定 {n} 件\n")
    p(f"{'':16s} {'採用可':>8s} {'採用率':>8s}   {'95%信頼区間':>16s}")
    for name, kk in ((a, ka), (b, kb), (f"{a}+{b} の2案", both)):
        lo, hi = wilson(kk, n)
        p(f"{name:16s} {kk:6d}/{n:<3d} {kk/n*100:7.1f}%   "
          f"[{lo*100:5.1f}%, {hi*100:5.1f}%]")

    p(f"\n内訳: 両方可 {n_ab} / {a} だけ {only_a} / {b} だけ {only_b} / 両方ダメ {none}")

    # 対応のある比較は McNemar (不一致ペアだけを使う)
    d = only_a + only_b
    if d:
        pv = min(1.0, 2 * sum(math.comb(d, i) * 0.5 ** d
                              for i in range(min(only_a, only_b) + 1)))
        p(f"\n対応のある比較 (McNemar 正確検定、不一致 {d} 件):")
        p(f"  {a} だけ可 {only_a} 対 {b} だけ可 {only_b}   p = {pv:.4f}")
        if pv >= 0.05:
            p(f"  -> 採用率に差があるとは言えない")
        else:
            p(f"  -> **{a if only_a > only_b else b} の採用率が有意に高い**")
    else:
        p("\n不一致ペアが無い (2 手法の可否が完全に一致)")

    p(f"\n2案にする価値: {a} 単独 {ka/n*100:.1f}% -> 2案 {both/n*100:.1f}% "
      f"(+{(both-ka)/n*100:.1f} 点)")
    p(f"                {b} 単独 {kb/n*100:.1f}% -> 2案 {both/n*100:.1f}% "
      f"(+{(both-kb)/n*100:.1f} 点)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--a", default="lama")
    ap.add_argument("--b", default="yoeo")
    ap.add_argument("--bench", default="cache/bench_removal.npz")
    ap.add_argument("--out", default="out/accept")
    ap.add_argument("--cache-dir", default="cache/core_out")
    ap.add_argument("--seed", type=int, default=13)
    x = ap.parse_args()

    if x.build:
        build(x.bench, x.out, x.a, x.b, x.seed, x.cache_dir)
    elif x.tally:
        tally(x.out)
    else:
        ap.error("--build か --tally を指定すること")
    return 0


if __name__ == "__main__":
    sys.exit(main())
