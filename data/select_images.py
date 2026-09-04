"""Day 3 / 層別選抜 — タイル化する 6,000枚を選ぶ (SPEC §6.4)。

入力: cache/index.jsonl, cache/score_raw.jsonl, cache/bursts.jsonl
出力: cache/selection.jsonl

**なぜショット単位の均等割りなのか**

枚数比例で選ぶとポートレート撮影が支配的になる。実測で上位ショットは
622 / 552 / 540 / 536 枚といずれも人物撮影で、風景・建築・寺社のテクスチャ多様性が薄まる。
一方、海外撮影分は「1日1都市 = 1ショット」の粒度なのでショット数は多いが枚数は少ない。

ショット単位で均等に割り当てると、海外分の多様性が自動的に優遇され、
ポートレートの過剰代表が抑えられる。

**なぜ層内ランクで評価するのか**

`snr` などの指標は lag 6 = 生画素 6px で測るため、40MP の X-T5 と 24MP の X-E3 では
物理スケールが異なり、機種間の絶対比較ができない。また `snr` の分布は
p50=43 に対し p99=17191 と極端な裾を持つ。
したがって**ショット内の順位**だけを使い、絶対値は閾値判定にのみ用いる。

**役割タグ**

  detail    : Stage C 用の高精細ソース (低ISO・高 snr)
  diversity : Stage A/B 用。対応場の学習は被写体ドメイン依存が弱いので数を稼ぐ

  pixi run python data/select_images.py
  pixi run python data/select_images.py --target 6000 --cap 40
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

INDEX = "cache/index.jsonl"
SCORES = "cache/score_raw.jsonl"
BURSTS = "cache/bursts.jsonl"
OUT = "cache/selection.jsonl"

# ハードゲート (絶対値で判定する。層内ランクでは救えない失敗)
MAX_CLIP = 0.05          # 飽和 5% 超は白飛びしすぎ
MIN_P999 = 3000          # 極端なアンダー
MIN_TEXTURE = 0.20       # 空や無地の壁だけの画像
MIN_SNR = 5.0            # 分布の p10 が 6.7。最下位帯を落とす
MAX_ANISO = 0.25         # 方向性ブレ (手ブレ)

# 役割判定
DETAIL_ISO = 400
DETAIL_SNR_RANK = 0.5    # ショット内で上位 50% の snr


def p(*a):
    print(*a, flush=True)


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def gate(r) -> str | None:
    """落とす理由を返す。通れば None。"""
    if "error" in r:
        return "error"
    if r["clip"] > MAX_CLIP:
        return "clip"
    if r["p999"] < MIN_P999:
        return "dark"
    if r["texture"] < MIN_TEXTURE:
        return "flat"
    if r["snr"] < MIN_SNR:
        return "snr"
    if r["aniso"] > MAX_ANISO:
        return "blur"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--bursts", default=BURSTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--target", type=int, default=6000)
    ap.add_argument("--cap", type=int, default=40, help="1ショットあたりの上限")
    a = ap.parse_args()

    for f in (a.index, a.scores):
        if not os.path.exists(f):
            p(f"入力がない: {f}")
            return 1

    scores = {r["path"]: r for r in load_jsonl(a.scores)}
    p(f"採点 {len(scores):,} 件")

    # --- 近重複を1枚に畳む ---
    dup_drop = set()
    if os.path.exists(a.bursts):
        for g in load_jsonl(a.bursts):
            paths = g.get("paths", [])
            if len(paths) > 1:
                # グループ内で snr 最良の1枚を残す
                best = max(paths, key=lambda q: scores.get(q, {}).get("snr", -1))
                dup_drop |= set(paths) - {best}
        p(f"近重複として除外 {len(dup_drop):,} 枚")

    # --- ハードゲート ---
    reasons = collections.Counter()
    pool = collections.defaultdict(list)
    for path, r in scores.items():
        if path in dup_drop:
            reasons["duplicate"] += 1
            continue
        why = gate(r)
        if why:
            reasons[why] += 1
            continue
        pool[(r.get("body"), r.get("shoot"))].append(r)

    n_pool = sum(len(v) for v in pool.values())
    p(f"\n=== ハードゲート ===")
    for k, v in reasons.most_common():
        p(f"  {k:<10} {v:>7,} 除外")
    p(f"  通過      {n_pool:>7,} 枚 / ショット {len(pool):,} 件")

    # --- ショット内で順位付け ---
    for key, rs in pool.items():
        n = len(rs)
        for metric, w in (("snr", 0.5), ("texture", 0.3), ("detail", 0.2)):
            order = np.argsort(np.argsort([r[metric] for r in rs]))
            for r, o in zip(rs, order):
                r.setdefault("_score", 0.0)
                r["_score"] += w * (o / max(n - 1, 1))
                if metric == "snr":
                    r["_snr_rank"] = o / max(n - 1, 1)
        # 低ISO を優遇 (絶対値で判定してよい数少ない指標)
        for r in rs:
            iso = r.get("iso") or 0
            r["_score"] += 0.15 if iso <= 200 else (0.10 if iso <= 400 else 0.0)
        rs.sort(key=lambda r: -r["_score"])

    # --- ショット単位の割り当て (水位法) ---
    quota = {k: 0 for k in pool}
    remaining = a.target
    while remaining > 0:
        open_shoots = [k for k in pool
                       if quota[k] < min(a.cap, len(pool[k]))]
        if not open_shoots:
            break
        share = max(remaining // len(open_shoots), 1)
        for k in open_shoots:
            if remaining <= 0:
                break
            room = min(a.cap, len(pool[k])) - quota[k]
            take = min(share, room, remaining)
            quota[k] += take
            remaining -= take

    sel = []
    for k, rs in pool.items():
        for r in rs[:quota[k]]:
            iso = r.get("iso") or 0
            role = ("detail" if iso <= DETAIL_ISO
                    and r.get("_snr_rank", 0) >= DETAIL_SNR_RANK else "diversity")
            sel.append(dict(
                path=r["path"], body=r.get("body"), shoot=r.get("shoot"),
                iso=iso, role=role,
                score=round(r["_score"], 4), snr=r["snr"],
                texture=r["texture"], detail=r["detail"], noise=r["noise"],
                clip=r["clip"], aniso=r["aniso"]))

    with open(a.out, "w") as fo:
        for r in sorted(sel, key=lambda d: (d["body"], d["shoot"], -d["score"])):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    p(f"\n=== 選抜 {len(sel):,} 枚 -> {a.out} ===")
    by_body = collections.Counter(r["body"] for r in sel)
    for b, c in sorted(by_body.items()):
        det = sum(1 for r in sel if r["body"] == b and r["role"] == "detail")
        p(f"  {b:<6} {c:>6,} 枚  (detail {det:,} / diversity {c-det:,})")
    role = collections.Counter(r["role"] for r in sel)
    p(f"  役割: detail {role['detail']:,} / diversity {role['diversity']:,}")
    q = collections.Counter(quota.values())
    p(f"  参加ショット {sum(1 for v in quota.values() if v):,} 件"
      f"  (1ショットあたり中央値 {int(np.median([v for v in quota.values() if v]))} 枚)")
    isos = collections.Counter("<=200" if r["iso"] <= 200 else
                               "201-400" if r["iso"] <= 400 else
                               "401-800" if r["iso"] <= 800 else ">800"
                               for r in sel)
    p(f"  ISO: " + "  ".join(f"{k} {v:,}" for k, v in sorted(isos.items())))
    est_tiles = len(sel) * 33
    p(f"\n  推定タイル数 {est_tiles:,} (1枚33タイル) / 目標 20万")
    return 0


if __name__ == "__main__":
    sys.exit(main())
