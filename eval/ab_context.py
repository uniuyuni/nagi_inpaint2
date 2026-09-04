"""文脈の広さの盲検A/B: 512原寸 対 1024→512 (docs/DESIGN_V2_JA.md §7)。

**問い**: 除去コアに見せる視野を 4 倍にすると (その代わり解像度が半分になる)、
512 原寸より良くなるか。

    A/B のどちらか  512²  をそのままコアへ            視野 基準 / 解像度 原寸
    A/B のもう一方  1024² を 512 へ縮小してコアへ      視野 4倍 / 解像度 半分
                    出力を 1024 へ戻し、HF 移植 + Poisson 合成

**初版は判定不能だった (ユーザ指摘 2026-08-30)**

「A/B で表示されている箇所が違うので判定できません」。
広視野側の出力を **1024 のまま**並べていたため、
元画像・マスク・512原寸 の 3 枚は native 領域を、
広視野の 1 枚だけが 4 倍の領域を写していた。**別の場所を見比べさせていた。**

**修正**: 広視野側も 1024 で合成まで済ませたあと、
**512原寸とまったく同じ native 領域を切り出して**並べる。
4 コマすべてが同じ場所・同じ画角・同じ画素数になる。
広視野の利点 (周りをより多く見た) は結果の中身に残り、
不利 (穴の中の解像度が半分) もそのまま残るので、比較の意味は変わらない。

**2 つのベンチが同じ場所を指していることの実測** (60 件すべて):

    native 領域の画素差   max 0.000000   <- ビット単位で一致
    マスクの IoU          中央値 1.000 / 最小 0.929

ずれの原因は検出マスクの拡大丸めなので、**広視野側のマスクは
native のマスクを 1024 の台紙に貼り直したものを使う** (完全に同一の対象を消す)。

**A/B の割り当ては厳密に半々** (R20)。コイン投げだと 20 件で 5/15 の偏りが出る
(実際に出た)。判定者に左右の癖があると結果が歪む。

  pixi run python eval/ab_context.py --build
  # out/ab_ctx/*.png を見て out/ab_ctx/votes.txt に A/B/o/x を書く
  pixi run python eval/ab_context.py --tally
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


def load_pairs(bench_n: str, bench_w: str):
    """2 つのベンチを突き合わせ、native 領域の位置 (dy,dx) を添えて返す。"""
    a = np.load(bench_n, allow_pickle=False)
    b = np.load(bench_w, allow_pickle=False)
    ma = json.loads(str(a["meta"]))
    mb = json.loads(str(b["meta"]))
    C, Cw = int(a["crop"]), int(b["crop"])
    out = []
    for i, (ra, rb) in enumerate(zip(ma, mb)):
        if ra["path"] != rb["path"]:
            continue
        dy, dx = ra["y0"] - rb["y0"], ra["x0"] - rb["x0"]
        if not (0 <= dy <= Cw - C and 0 <= dx <= Cw - C):
            continue
        out.append(dict(i=i, name=f"{i:02d}_{os.path.basename(ra['path'])}",
                        img_n=a["img"][i].astype(np.float32),
                        msk_n=a["mask"][i].astype(bool),
                        img_w=b["img"][i].astype(np.float32),
                        dy=dy, dx=dx, C=C, Cw=Cw))
    return out


def build(bench_n: str, bench_w: str, out: str, seed: int, device: str,
          n: int) -> None:
    from nagi_inpaint2.pipeline import Inpainter
    from nagi_inpaint2.removers import composite

    cases = load_pairs(bench_n, bench_w)
    p(f"{len(cases)} 件が 2 つのベンチで突き合った")

    # 判定の手間を抑えるため、**両者の出力差が大きい順**に n 件へ絞る。
    # この基準は 512原寸 と 広視野 の間で完全に対称 (R20)。
    inp = Inpainter(core="yoeo", device=device, crop=512, detail="hf",
                    poisson=True, steps=2)
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()

    for k, c in enumerate(cases):
        C, Cw, dy, dx = c["C"], c["Cw"], c["dy"], c["dx"]
        img_n, m_n = c["img_n"], c["msk_n"]

        # --- 広視野側のマスクは native のマスクを貼り直す (同一の対象) ---
        m_w = np.zeros((Cw, Cw), bool)
        m_w[dy:dy + C, dx:dx + C] = m_n

        # --- 実際の製品コードの経路をそのまま通す ---
        o_nat = composite(inp.fill_crop(img_n, m_n), img_n, m_n, poisson=True)
        o_nat[~m_n] = img_n[~m_n]

        o_w = composite(inp.fill_crop(c["img_w"], m_w), c["img_w"], m_w,
                        poisson=True)
        o_w[~m_w] = c["img_w"][~m_w]
        # **native とまったく同じ領域を切り出す。** ここが初版の欠陥の修正点。
        o_wide = o_w[dy:dy + C, dx:dx + C]

        c["o_nat"], c["o_wide"] = o_nat, o_wide
        c["diff"] = float(np.abs(o_nat[m_n] - o_wide[m_n]).mean())
        if (k + 1) % 10 == 0:
            el = time.perf_counter() - t0
            p(f"  {k+1}/{len(cases)}  {el/(k+1):.1f} 秒/件  "
              f"残り約 {(len(cases)-k-1)*el/(k+1)/60:.0f} 分")

    cases.sort(key=lambda c: -c["diff"])
    if 0 < n < len(cases):
        cases = cases[:n]
        p(f"\n出力差が大きい上位 {len(cases)} 件に絞った (両者に対称な基準)")
        p(f"  差の範囲 {cases[-1]['diff']:.4f} 〜 {cases[0]['diff']:.4f}")

    # --- A/B を厳密に半々に割り当てる (R20) ---
    rng = random.Random(seed)
    flags = [True] * (len(cases) // 2) + [False] * (len(cases) - len(cases) // 2)
    rng.shuffle(flags)

    key_map: dict[str, str] = {}
    for rank, (c, wide_is_A) in enumerate(zip(cases, flags)):
        A, B = ((c["o_wide"], c["o_nat"]) if wide_is_A
                else (c["o_nat"], c["o_wide"]))
        key_map[c["name"]] = "A" if wide_is_A else "B"

        img = c["img_n"]
        ov = img.copy()
        ov[c["msk_n"]] = ov[c["msk_n"]] * 0.4 + np.array([1.0, .25, .25]) * 0.6
        panels = (img, ov, A, B)
        H, W = img.shape[:2]
        sheet = np.full((H, W * 4 + 12, 3), 32, np.uint8)
        for j, pan in enumerate(panels):
            sheet[:, j * (W + 4):j * (W + 4) + W] = u8(pan)
        Image.fromarray(sheet).save(
            os.path.join(out, f"{rank:02d}_{c['name']}.png"))

    json.dump({"cases": key_map, "seed": seed,
               "bench": [bench_n, bench_w]},
              open(os.path.join(out, "key.json"), "w"))
    with open(os.path.join(out, "votes.txt"), "w") as f:
        f.write("# 各行: <ケース名> <A|B|o|x>\n"
                "# 列は 左から: 元画像 | マスク(赤) | A | B\n"
                "# **4 コマとも同じ場所・同じ画角**。A と B は処理の違いだけ\n"
                "# 観点: 破綻がないか / つながっているか / 消えているか\n"
                "#   o = どちらも良い / x = どちらもダメ\n")
        for nm in key_map:
            f.write(f"{nm} \n")
    p(f"\n{len(key_map)} 件 -> {out}/")
    wa = sum(1 for v in key_map.values() if v == "A")
    p(f"  広視野が A: {wa}/{len(key_map)} / B: {len(key_map)-wa}")
    p(f"  済んだら: pixi run python eval/ab_context.py --tally")


def tally(out: str) -> None:
    key = json.load(open(os.path.join(out, "key.json")))["cases"]
    votes: dict[str, str] = {}
    for line in open(os.path.join(out, "votes.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        if len(t) > 1:
            votes[t[0]] = t[1].upper()

    win = lose = good = bad = 0
    for nm, v in votes.items():
        if nm not in key:
            continue
        if v in ("O", "○", "0"):
            good += 1
        elif v in ("X", "×"):
            bad += 1
        elif v in ("A", "B"):
            win, lose = (win + 1, lose) if v == key[nm] else (win, lose + 1)
    n = win + lose
    tot = n + good + bad
    p(f"投票 {len(votes)} 件 / 勝敗つき {n} 件")
    p(f"  どちらも良い {good} 件 / どちらもダメ {bad} 件"
      + (f"  (どちらもダメ = {bad}/{tot} = {bad/tot*100:.0f}%)" if tot else ""))
    if not n:
        p("\n判定なし")
        return
    pv = sum(math.comb(n, i) * 0.5 ** n
             for i in range(min(win, lose) + 1)) * 2
    p(f"\n=== 広視野 (1024→512) 対 512原寸 ===")
    p(f"  広視野の勝ち {win} / 負け {lose}  勝率 {win/n*100:.1f}% (n={n})")
    p(f"  両側二項検定 p = {min(pv,1.0):.4f}")
    sig = pv < 0.05
    if win > lose:
        p("  -> 広視野を採用" if sig else "  -> 広視野が優勢だが有意ではない (件数不足)")
    elif sig:
        p("  -> **広視野は有意に悪い。** 512原寸を維持する")
    else:
        p("  -> 差は出ていない。512原寸のままで良い")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--bench", default="cache/bench_removal.npz")
    ap.add_argument("--bench-wide", default="cache/bench_removal_1024.npz")
    ap.add_argument("--out", default="out/ab_ctx")
    ap.add_argument("--n", type=int, default=20, help="0 なら全件")
    ap.add_argument("--seed", type=int, default=57)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    if a.build:
        build(a.bench, a.bench_wide, a.out, a.seed, a.device, a.n)
    elif a.tally:
        tally(a.out)
    else:
        ap.error("--build か --tally を指定すること")
    return 0


if __name__ == "__main__":
    sys.exit(main())
