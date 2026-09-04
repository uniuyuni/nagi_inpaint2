"""Day 3 / バースト検出と擬似ペア発掘 (SPEC §6.4 の重複排除, §9.2 の評価データ)。

cache/hash.jsonl だけを読む。RAW にも JPEG にも触らないので、
他のパスと並行して実行できる。

**2つの出力**

1. `cache/bursts.jsonl` — 近重複のグループ。
   連写やブラケットで実質同じ画像を複数選抜しないために select.py が使う。

2. `cache/pairs.jsonl` — 擬似ペア候補 (§9.2)。
   評価用のペア撮影データが存在しないため、「同じ構図で被写体だけが動いた前後」を
   既存ライブラリから掘り出す。三脚ペア撮影と実質同等の評価データになる。

**v1 の失敗 (目視検証で判明)**

「変化セルの割合 diff_cells が大きいほど良い候補」としたところ、上位8組すべてが
使い物にならなかった:

  - 露出と時刻の違い (夕景の明るさ変化、街灯の点灯)
  - 構図のずれ (画角・立ち位置の変更)
  - そもそも別構図

diff_cells は**全画面的な変化**でも大きくなる。欲しいのは
「大部分は完全に一致していて、一部だけが違う」という**局所性**であって、
変化量の大きさではなかった。

**v2 の判別器**

16x16 の輝度サムネイル (z 正規化済み) の差分マップに対して:

  shift_gain : ±2 セルの平行移動で相関がどれだけ改善するか。
               改善が大きい = 構図がずれているだけ -> 棄却
  compact    : 変化セルの最大連結成分 / 変化セル総数。
               1.0 に近い = 変化が1箇所に固まっている = 被写体の有無
  bg_corr    : 変化セルを除いた領域の相関。
               1.0 に近い = 背景が完全に一致 = 三脚相当の安定性

  擬似ペア: shift_gain < 0.02 かつ compact > 0.6 かつ bg_corr > 0.97
            かつ 0.01 < diff_cells < 0.20

**限界**: 16x16 は粗いので、フレームの 1/16 より小さい変化は検出できない。
**候補生成器であって最終判定ではない**。採用前に必ず目視で確認すること
(v1 の失敗がまさにその実例)。

探索は同一ショットフォルダ内に閉じる (同一フォルダ = 同一ショット)。

  pixi run python data/pairs.py
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import sys

import numpy as np

HASHES = "cache/hash.jsonl"
OUT_BURST = "cache/bursts.jsonl"
OUT_PAIRS = "cache/pairs.jsonl"

N = 16               # サムネイルの辺
DUP_CORR = 0.98
DUP_CELLS = 0.01
PAIR_CORR = 0.90
CELLS_LO, CELLS_HI = 0.01, 0.20
CELL_THR = 0.8       # z 正規化サムネイルで「変化した」とみなす差
MAX_SHIFT_GAIN = 0.02
MIN_COMPACT = 0.60
MIN_BG_CORR = 0.97
MAX_SHOOT = 800


def p(*a):
    print(*a, flush=True)


def parse_dt(s):
    if not s:
        return 0.0
    try:
        return dt.datetime.strptime(s, "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return 0.0


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def largest_component(mask: np.ndarray) -> int:
    """4近傍の最大連結成分のセル数。"""
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack, size = [(sy, sx)], 0
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            best = max(best, size)
    return best


def pair_features(ta: np.ndarray, tb: np.ndarray) -> dict:
    """16x16 サムネイル 2枚から判別特徴を出す。"""
    A, B = ta.reshape(N, N), tb.reshape(N, N)
    base = float((A * B).mean())

    # 平行移動で相関が改善するなら、それは構図のずれ
    best_shift = base
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            if dy == 0 and dx == 0:
                continue
            a = A[max(dy, 0):N + min(dy, 0), max(dx, 0):N + min(dx, 0)]
            b = B[max(-dy, 0):N + min(-dy, 0), max(-dx, 0):N + min(-dx, 0)]
            if a.size:
                best_shift = max(best_shift, float((a * b).mean()))
    shift_gain = best_shift - base

    d = np.abs(A - B)
    mask = d > CELL_THR
    n_changed = int(mask.sum())
    cells = n_changed / (N * N)

    if n_changed:
        compact = largest_component(mask) / n_changed
        keep = ~mask
        bg_corr = float((A[keep] * B[keep]).mean()) if keep.sum() > 16 else 0.0
    else:
        compact, bg_corr = 0.0, base

    return dict(corr=round(base, 4), diff_cells=round(cells, 4),
                shift_gain=round(shift_gain, 4),
                compact=round(compact, 4), bg_corr=round(bg_corr, 4))


def analyse_shoot(recs):
    n = len(recs)
    T = np.array([r["tiny"] for r in recs], dtype=np.float32)
    corr = (T @ T.T) / T.shape[1]

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    pairs = []
    for i in range(n):
        for j in (np.nonzero(corr[i, i + 1:] > PAIR_CORR)[0] + i + 1):
            j = int(j)
            f = pair_features(T[i], T[j])
            if f["corr"] > DUP_CORR and f["diff_cells"] < DUP_CELLS:
                union(i, j)
                continue
            if not (CELLS_LO < f["diff_cells"] < CELLS_HI):
                continue
            if f["shift_gain"] >= MAX_SHIFT_GAIN:      # 構図ずれ
                continue
            if f["compact"] < MIN_COMPACT:             # 変化が散在 = 光線変化など
                continue
            if f["bg_corr"] < MIN_BG_CORR:             # 背景が一致していない
                continue
            f.update(a=recs[i]["path"], b=recs[j]["path"],
                     dhash_dist=hamming(recs[i]["dhash"], recs[j]["dhash"]),
                     dt_sec=abs(parse_dt(recs[i].get("datetime"))
                                - parse_dt(recs[j].get("datetime"))),
                     body=recs[i].get("body"), shoot=recs[i].get("shoot"))
            pairs.append(f)

    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values()), pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hashes", default=HASHES)
    ap.add_argument("--bursts", default=OUT_BURST)
    ap.add_argument("--pairs", default=OUT_PAIRS)
    a = ap.parse_args()

    if not os.path.exists(a.hashes):
        p(f"ハッシュがない: {a.hashes}  先に data/score_jpeg.py を実行すること")
        return 1

    recs = []
    for line in open(a.hashes):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "tiny" in r and "dhash" in r:
            recs.append(r)
    p(f"ハッシュ {len(recs):,} 件")

    by = collections.defaultdict(list)
    for r in recs:
        by[(r.get("body"), r.get("shoot"))].append(r)
    p(f"ショット {len(by):,} 件\n")

    n_groups = n_dups = 0
    all_pairs = []
    with open(a.bursts, "w") as fb:
        for gi, (key, rs) in enumerate(sorted(by.items(), key=lambda kv: str(kv[0]))):
            if len(rs) > MAX_SHOOT:
                rs = rs[:MAX_SHOOT]
            rs.sort(key=lambda r: parse_dt(r.get("datetime")))
            groups, pairs = analyse_shoot(rs)
            for g in groups:
                if len(g) > 1:
                    n_dups += len(g) - 1
                n_groups += 1
                fb.write(json.dumps(dict(
                    body=key[0], shoot=key[1],
                    paths=[rs[i]["path"] for i in g]), ensure_ascii=False) + "\n")
            all_pairs += pairs
            if (gi + 1) % 50 == 0:
                p(f"  {gi+1}/{len(by)} ショット  重複 {n_dups:,}  候補 {len(all_pairs):,}")

    all_pairs.sort(key=lambda d: (-d["compact"], -d["bg_corr"]))
    with open(a.pairs, "w") as fp:
        for d in all_pairs:
            fp.write(json.dumps(d, ensure_ascii=False) + "\n")

    p(f"\n=== 近重複 ===")
    p(f"  グループ {n_groups:,} / 除ける枚数 {n_dups:,} "
      f"({n_dups/max(len(recs),1)*100:.1f}%)  -> {a.bursts}")
    p(f"\n=== 擬似ペア候補 (§9.2) ===")
    p(f"  {len(all_pairs):,} 組 / ショット {len(set(d['shoot'] for d in all_pairs)):,} 件")
    for d in all_pairs[:8]:
        p(f"    compact {d['compact']:.2f} bg {d['bg_corr']:.3f} "
          f"diff {d['diff_cells']:.3f} shift {d['shift_gain']:+.4f} "
          f"Δt {d['dt_sec']:>6.0f}s  {os.path.basename(d['a'])}<->"
          f"{os.path.basename(d['b'])} [{d['shoot']}]")
    p(f"  -> {a.pairs}")
    p("\n目視確認が必須。data/pairs_sheet.py でコンタクトシートを作ること。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
