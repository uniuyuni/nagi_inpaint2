"""盲検A/B の共通部分 — 無作為化と集計 (docs/DESIGN_V2_JA.md §6)。

元は `eval/ab_compare.py` にあったが、あのファイルは v1 の生成器
(`stage_b_search` / `stage_f`) を import する v1 専用のスクリプトだった。
**v1 の生成器は 0勝16敗で決着済み**なので消したが、
無作為化と集計はどの比較でも使う (`ab_removal` / `ab3` など) ので、
依存の無いモジュールとしてここへ移した (2026-09-04)。

`assign_sides` は 5 seed で偏りが無いことを `tests/test_ab.py` が実測している。

> **同点の書き方に注意 (2026-08-25 に発覚)**: `votes.txt` には `=` が
> 書かれていたのに `tally_votes` は "同等" しか見ておらず、
> 同点が 0 件と表示されていた (分母からは正しく除かれていた)。
> 別名を全部受けるようにしてある。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np


def assign_sides(case_names: list[str], seed: int) -> dict[str, bool]:
    """ケースごとに「我々が A か」を無作為に決める。

    左右の無作為化が偏らないこと・seed が同じなら再現できることの両方を、
    `tests/test_ab.py` が実測で確認する。1ケースにつき1回のコイン投げに
    独立した RNG を使い、ケースの追加順に依存しない安定な割り当てにする
    (`np.random.default_rng(seed)` を1本のストリームとして順に消費する)。
    """
    rng = np.random.default_rng(seed)
    return {name: bool(rng.random() < 0.5) for name in case_names}


def binomial_test_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """両側二項検定の p 値を自前計算する (scipy 依存を増やさないため)。

    帰無仮説: 各試行が確率 p0 で成功する。観測 k 勝 n 戦がどれだけ極端かを、
    「観測した確率質量以下のすべての結果」を足し上げる標準的な定義で測る。
    p0=0.5 では二項分布が対称なので、`math.comb` で愚直に列挙しても
    n が小さい (盲検比較は多くて数十ケース) 前提では十分速い。
    """
    if n == 0:
        return float("nan")
    obs = math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
        if pi <= obs * (1 + 1e-9):
            total += pi
    return float(min(1.0, total))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval (既定 z はその臨界値)。n=0 なら (nan, nan)。"""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def tally_votes(votes: dict[str, str], key_map: dict[str, str]) -> dict:
    """`votes.txt` の判定 (A/B/同等) を `key.json` (どちらが我々か) と突き合わせる。

    戻り値: wins/losses/ties の件数、我々の勝率、95%信頼区間 (Wilson)、
    二項検定 p 値 (「同等」は除外)。
    """
    wins = losses = ties = 0
    for name, vote in votes.items():
        if name not in key_map:
            continue
        ours_side = key_map[name]
        # 記入者が使う表記の揺れを吸収する。**全角の `＝` を忘れていた**
        # (半角 `=` が 2 回書かれているだけだった。2026-09-04 に発見)。
        # 日本語入力のまま打つと全角になるので実際に起こりうる。
        if vote in ("同等", "=", "＝", "tie", "TIE"):
            ties += 1
        elif vote == ours_side:
            wins += 1
        elif vote in ("A", "B"):
            losses += 1
    n = wins + losses
    winrate = wins / n if n else float("nan")
    lo, hi = wilson_interval(wins, n)
    pval = binomial_test_two_sided(wins, n)
    return dict(wins=wins, losses=losses, ties=ties, n=n, winrate=winrate,
               ci_lo=lo, ci_hi=hi, pvalue=pval)


def parse_votes_file(path: str) -> dict[str, str]:
    votes: dict[str, str] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, vote = parts[0], parts[1]
        votes[name] = vote
    return votes


def tally(out: str) -> dict:
    key_map = json.load(open(os.path.join(out, "key.json")))["cases"]
    votes = parse_votes_file(os.path.join(out, "votes.txt"))
    result = tally_votes(votes, key_map)

    p(f"投票 {len(votes)} 件 / ケース {len(key_map)} 件")
    p(f"我々の勝ち {result['wins']}  負け {result['losses']}  同等 {result['ties']}")
    if result["n"]:
        p(f"勝率 {result['winrate']*100:.1f}%  "
          f"95%CI [{result['ci_lo']*100:.1f}, {result['ci_hi']*100:.1f}]%  "
          f"(n={result['n']}, 同等除く)")
        p(f"両側二項検定 p = {result['pvalue']:.5f}")
    else:
        p("勝敗の付いた投票が無い (votes.txt を確認すること)")
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# --- 対応のある比較の統計 (元 eval/benchmark.py。あちらは v1 の生成器を引くので移した) ---

def bootstrap_median_ci(diffs: list[float], n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float]:
    """対応のある差の中央値のブートストラップ 95%信頼区間。

    **なぜ要るか**: 人物ベンチ 60ケースでの 0.0059 対 0.0052 という差が
    統計的に有意かを一度も確認していなかった。中央値だけを並べていると、
    雑音を改善だと誤読する事故が起きる。ケースを重複を許して再標本し、
    その中央値の分布から区間を作ることで、この差がどれだけ揺れうるかを見る。
    """
    v = np.asarray([d for d in diffs if np.isfinite(d)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.median(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def sign_test_pvalue(diffs: list[float]) -> float:
    """符号検定の両側 p 値 (0 との差)。scipy を増やさず `math.comb` で自前計算する。

    「同点 (差が厳密に0)」は検定から除外する (符号検定の標準的な扱い)。
    """
    v = [d for d in diffs if np.isfinite(d) and d != 0]
    n = len(v)
    if n == 0:
        return float("nan")
    k = sum(1 for d in v if d > 0)
    obs = math.comb(n, k) * (0.5 ** n)
    total = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (0.5 ** n)
        if pi <= obs * (1 + 1e-9):
            total += pi
    return float(min(1.0, total))


def paired_comparison(ours: list[dict], base: list[dict], key: str) -> dict:
    """`ours[i][key] - base[i][key]` (ケースごとの対応する差) の要約統計を返す。

    `ours` と `base` は同じケース列を同じ順序で処理した結果 (`run()` 内では
    どちらも `spec["cases"]` を同一順で辿って積んでいるので、インデックスが
    そのままケース対応になる)。戻り値: `n`(有限差の件数), `median`,
    `ci_lo`/`ci_hi` (ブートストラップ2000回), `pvalue` (符号検定)。
    """
    diffs = [a.get(key, float("nan")) - b.get(key, float("nan"))
             for a, b in zip(ours, base)]
    diffs = [d for d in diffs if np.isfinite(d)]
    lo, hi = bootstrap_median_ci(diffs)
    return dict(n=len(diffs),
               median=float(np.median(diffs)) if diffs else float("nan"),
               ci_lo=lo, ci_hi=hi, pvalue=sign_test_pvalue(diffs))
