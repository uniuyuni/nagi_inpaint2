"""実除去ベンチでの盲検A/B (docs/DESIGN_V2_JA.md §6)。

**なぜこれが合否を決めるのか**

v1 は「簡易 LPIPS で 13% 劣位＝接近している」と判断し、人手では 0勝16敗だった。
代理指標が人間の判断を代表しなかったのが敗因である (§6.1)。
今回も自動指標は当てにならないことが実測で分かっている:
陰性対照が 20/29 しか通らず、正当な場面継続を「新規物体」と誤検出した。

**したがって合否は人手の盲検A/Bだけで決める。**

無作為化・二項検定・Wilson 区間は `eval/ab_compare.py` の実装をそのまま使う
(`assign_sides` は 5 seed で偏りが無いことを `tests/test_ab.py` が実測済み)。

**判定の観点** (ユーザ指示 2026-08-20):
  1. 破綻がないか   2. うまくつながっているか   3. 消したいものが消えているか

**件数を絞るときの原則** (ユーザ要望 2026-08-25「境界が見えるものに絞れない?」)

差が見えないケースは同等票になるだけで情報を持たない (前回 60 件中 17 件が同等)。
絞ること自体は正しいが、**片方に有利な選び方をしてはいけない** —
それは v1 が 0勝16敗した構造 (代理指標で判断する) の再現になる。

**max を使う案は失敗した (2026-08-25 に実測で棄却)**

    score = max(bd(LaMa), bd(YOEO))   -> 採用24件で「LaMa の方が継ぎ目が強い」は 1/24

YOEO の継ぎ目が系統的に強い (中央値 2.09 対 1.44) ため、max はほぼ常に YOEO 側を拾い、
**「YOEO の継ぎ目が最悪なケース」だけを集めてしまう**。対称に見えて対称ではない。

**採用する方式: 各手法の「継ぎ目が強い順」から半数ずつ取る。**
LaMa の worst N/2 と YOEO の worst N/2 を合わせる (重複は次点で埋める)。
これなら「なぜこのケースが選ばれたか」の理由が両手法に均等に配分される。
`boundary_discontinuity` は 1.0 が「継ぎ目が見えない」で、
`eval/metrics.py` の実装をそのまま使う (陰性対照テスト済み)。

**この絞り込みは測定の意味を変える。** 得られる勝率は
「全ケースでの勝率」ではなく「**継ぎ目が見えるケースでの勝率**」である。
報告のときに必ず明記すること。

  pixi run python eval/ab_removal.py --build
  # out/ab_removal/*.png を見て out/ab_removal/votes.txt に A/B/= を書く
  pixi run python eval/ab_removal.py --tally
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.ab_compare import assign_sides, tally                   # noqa: E402


def p(*a):
    print(*a, flush=True)


def u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def build(bench: str, out: str, seed: int, device: str, n: int,
          visible: int = 0, l5: bool = True, by: str = "diff") -> None:
    from eval.lama_baseline import Lama
    from eval.metrics import boundary_discontinuity
    from nagi_inpaint2.removers import Yoeo, composite

    z = np.load(bench, allow_pickle=False)
    imgs, msks = z["img"], z["mask"]
    meta = json.loads(str(z["meta"]))
    n = len(meta) if n <= 0 else min(n, len(meta))
    all_names = [f"{i:02d}_{os.path.basename(meta[i]['path'])}" for i in range(n)]

    lama = Lama(device=device)
    yoeo = Yoeo(device=device, steps=2)
    os.makedirs(out, exist_ok=True)

    # --- 全件を一度作り、対称な指標で採点する ---------------------------------
    cases = []
    for i in range(n):
        img = imgs[i].astype(np.float32)
        mask = msks[i].astype(bool)
        # L5 (Poisson) を**両者に**掛ける。LaMa には実質無効なので公平。
        ol = composite(lama(img, mask), img, mask, poisson=l5)
        oy = composite(yoeo(img, mask), img, mask, poisson=l5)
        bl = boundary_discontinuity(ol, mask)
        by = boundary_discontinuity(oy, mask)
        score = float(np.nanmax([bl, by])) if np.isfinite([bl, by]).any() else 0.0
        diff = float(np.abs(ol[mask] - oy[mask]).mean())
        cases.append(dict(i=i, name=all_names[i], img=img, mask=mask, ol=ol, oy=oy,
                          bd_lama=bl, bd_yoeo=by, score=score, diff=diff))
        p(f"  {i+1}/{n}  {all_names[i]}  継ぎ目 LaMa {bl:.2f} / YOEO {by:.2f}")

    if visible > 0 and by == "diff":
        # **両者の出力がどれだけ違うか**で選ぶ。完全に対称で、
        # 「見比べて判断できるケース」だけが残る (同じなら同等票にしかならない)。
        cases.sort(key=lambda c: -c["diff"])
        cases = cases[:visible]
        nl = sum(1 for c in cases if c["bd_lama"] > c["bd_yoeo"])
        p(f"\n両者の差が大きい上位 {len(cases)} 件に絞った (完全に対称な基準)")
        p(f"  参考: この中で LaMa の方が継ぎ目が強いのは {nl}/{len(cases)} 件")
    elif visible > 0:
        # **各手法の worst を半数ずつ**取る。max で選ぶと偏る (docstring 参照)。
        half = visible // 2
        by_l = sorted(cases, key=lambda c: -(c["bd_lama"] if np.isfinite(c["bd_lama"]) else 0))
        by_y = sorted(cases, key=lambda c: -(c["bd_yoeo"] if np.isfinite(c["bd_yoeo"]) else 0))
        chosen: dict[int, dict] = {}
        li = yi = 0
        while len(chosen) < visible and (li < len(by_l) or yi < len(by_y)):
            if len(chosen) < visible and li < len(by_l):
                c = by_l[li]; li += 1
                chosen.setdefault(c["i"], c)
            if len(chosen) < visible and yi < len(by_y):
                c = by_y[yi]; yi += 1
                chosen.setdefault(c["i"], c)
        cases = list(chosen.values())
        nl = sum(1 for c in cases if c["bd_lama"] > c["bd_yoeo"])
        p(f"\n継ぎ目が見える {len(cases)} 件に絞った (各手法の worst を半数ずつ)")
        p(f"  LaMa の方が継ぎ目が強い: {nl}/{len(cases)} 件 "
          f"(50% 付近なら選別が偏っていない)")
        cases.sort(key=lambda c: -max(c["bd_lama"], c["bd_yoeo"]))

    names = [c["name"] for c in cases]
    sides = assign_sides(names, seed)          # True なら YOEO が A
    key_map: dict[str, str] = {}

    for rank, c in enumerate(cases):
        yoeo_is_a = sides[c["name"]]
        A, B = (c["oy"], c["ol"]) if yoeo_is_a else (c["ol"], c["oy"])
        key_map[c["name"]] = "A" if yoeo_is_a else "B"
        img = c["img"]
        H, W = img.shape[:2]
        # 元画像 (消す前) も並べる。「消したいものが消えているか」を判定するため。
        sheet = np.full((H, W * 3 + 8, 3), 32, np.uint8)
        for j, pan in enumerate((img, A, B)):
            sheet[:, j * (W + 4):j * (W + 4) + W] = u8(pan)
        # 継ぎ目が見える順に並ぶよう、通し番号を頭に付ける
        Image.fromarray(sheet).save(
            os.path.join(out, f"{rank:02d}_{c['name']}.png"))

    json.dump([{k: v for k, v in c.items()
                if k in ("i", "name", "bd_lama", "bd_yoeo", "score", "diff")}
               for c in cases],
              open(os.path.join(out, "scores.json"), "w"),
              ensure_ascii=False, indent=1)

    # `eval/ab_compare.tally` が読む形式に合わせる ({"cases": {...}})
    json.dump({"cases": key_map, "seed": seed, "bench": bench},
              open(os.path.join(out, "key.json"), "w"))
    with open(os.path.join(out, "votes.txt"), "w") as f:
        f.write("# 各行: <ケース名> <A|B|=>\n"
                "# 列は 左:元画像(消す前) | 中央:A | 右:B\n"
                "# 観点: 破綻がないか / つながっているか / 消したいものが消えているか\n"
                "# どちらとも言えない場合は = と書く\n")
        for nm in names:
            f.write(f"{nm} \n")
    p(f"\n{len(names)} 件 -> {out}/")
    p(f"  画像を見て {out}/votes.txt に A / B / = を書き込む")
    p(f"  左が元画像 (消す前)、中央が A、右が B。**手法名は伏せてある**")
    p(f"  済んだら: pixi run python eval/ab_removal.py --tally")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--bench", default="cache/bench_removal.npz")
    ap.add_argument("--out", default="out/ab_removal")
    ap.add_argument("--n", type=int, default=0, help="0 なら全件")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-l5", action="store_true", help="L5 (Poisson) を切る")
    ap.add_argument("--by", choices=("diff", "seam"), default="diff",
                    help="絞り込みの基準。diff=両者の差 (対称) / seam=各手法の継ぎ目worst")
    ap.add_argument("--visible", type=int, default=0,
                    help=">0 なら「継ぎ目が見える」上位この件数に絞る。"
                         "score = max(LaMa, YOEO) で対称")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    if a.build:
        build(a.bench, a.out, a.seed, a.device, a.n, a.visible,
              l5=not a.no_l5, by=a.by)
    elif a.tally:
        r = tally(a.out)
        p(f"\n=== 盲検A/B: YOEO 対 LaMa (「我々」= YOEO) ===")
        p(f"  合格条件 (§7 Phase 0): 判定 40 件で 27 勝以上")
        if r["n"] < 40:
            p(f"  -> 判定 {r['n']} 件では検出力が足りない (40 件必要)")
        elif r["wins"] >= 27:
            p("  -> **合格** (YOEO が LaMa に勝ち越し)")
        else:
            p("  -> 不合格")
    else:
        ap.error("--build か --tally を指定すること")
    return 0


if __name__ == "__main__":
    sys.exit(main())
