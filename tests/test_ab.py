"""`eval/ab_stats.py` (盲検A/B の無作為化と集計) のテスト。

**なぜここだけ残したか**: 元の `tests/test_ab.py` は v1 の生成器を動かす
`ab_compare.build` も試していたが、v1 の生成器は 0勝16敗で決着し削除した。
**無作為化と集計はどの比較でも使う**ので、その部分だけを残す。
`assign_sides` が偏らないことは、v1 の頃からここで実測している。
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import ab_stats as ab                          # noqa: E402


def test_左右の割り当てが偏らない():
    names = [f"case_{i}" for i in range(200)]
    for seed in range(5):
        sides = ab.assign_sides(names, seed)
        a = sum(sides.values())
        # 200 件・p=0.5 の 3σ は約 21 件。これを超えたら偏っている
        assert abs(a - 100) < 21, f"seed={seed} で A 側 {a}/200"


def test_同じ_seed_なら再現する():
    names = [f"case_{i}" for i in range(50)]
    assert ab.assign_sides(names, 42) == ab.assign_sides(names, 42)
    assert ab.assign_sides(names, 42) != ab.assign_sides(names, 43)


def test_勝敗の集計():
    key = {"a": "A", "b": "B", "c": "A"}
    votes = {"a": "A", "b": "B", "c": "B"}      # 我々の 2勝1敗
    r = ab.tally_votes(votes, key)
    assert (r["wins"], r["losses"], r["n"]) == (2, 1, 3)


def test_同点は分母から除く():
    """**`=` が「同等」として数えられていなかった** (2026-08-25 に発覚)。
    分母からは正しく除かれていたが、表示上 0 件になっていた。"""
    key = {f"c{i}": "A" for i in range(4)}
    for tie in ("=", "＝", "同等", "tie", "TIE"):
        r = ab.tally_votes({"c0": "A", "c1": "B", "c2": tie, "c3": tie}, key)
        assert (r["wins"], r["losses"], r["n"]) == (1, 1, 2), f"{tie} が同点として扱われていない"
        assert r["ties"] == 2, f"{tie} が同点として数えられていない"


@pytest.mark.parametrize("k,n,expect", [(10, 10, 2 * 0.5 ** 10), (5, 5, 2 * 0.5 ** 5)])
def test_二項検定が既知の値に一致(k, n, expect):
    assert ab.binomial_test_two_sided(k, n) == pytest.approx(expect)


def test_二項検定は五分五分で_1_に近い():
    assert ab.binomial_test_two_sided(3, 5) == pytest.approx(1.0)


def test_wilson_区間が点推定を挟む():
    lo, hi = ab.wilson_interval(45, 60)
    assert lo < 45 / 60 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_符号検定と_bootstrap():
    assert ab.sign_test_pvalue([1.0] * 10) == pytest.approx(2 * 0.5 ** 10)
    assert math.isnan(ab.sign_test_pvalue([]))   # 標本ゼロは「判定なし」
    lo, hi = ab.bootstrap_median_ci([1.0] * 10, n_boot=500, seed=1)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert all(math.isnan(v) for v in ab.bootstrap_median_ci([]))
