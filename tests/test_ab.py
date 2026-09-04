"""`eval/ab_compare.py` のテスト。device は "cpu" 固定。

**なぜこれが要る (背景)**

このプロジェクトは指標に何度も裏切られてきた。
  - PSNR が LaMa より高い (14.1 vs 13.5 dB) のに、目視では明らかに劣る
    (白いぼけた塊が残る) ケースが複数あった。
  - 自作の簡易 LPIPS (VGG19 特徴距離、層は等重み、学習済み線形層なし) も
    同じ危険がある。
  - 人物ベンチ 60ケースでの 0.0059 対 0.0052 という差が**統計的に有意かを
    一度も確認していない**。

**効く理屈**: 最終的な品質判定を**人間の盲検比較**に置き換えれば、指標の
妥当性という不確実性が構造的に消える。加えて対比較の統計を出せば、雑音を
追いかける事故が防げる。

**効かない条件**: 提示順に偏りがあると判定が汚染される (常に右が我々、など)。
左右をケースごとに無作為化し、**判定が終わるまでどちらがどちらか分からない**
ようにすること。

このテストは以下を実測で担保する:
  1. `--build` が指定ケース数の PNG と `key.json`、`votes.txt` を作ること。
  2. 左右の無作為化が偏っていないこと (ours が A になる割合が 0.4〜0.6)。
  3. 同じ seed なら同じ割り当てになること (再現性)。
  4. `--tally` が正しく集計すること (自前実装の二項検定 p 値が既知の値と一致)。
  5. 対比較統計 (`eval.benchmark.paired_comparison`) が既知の入力で
     妥当な値を返すこと。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import ab_compare as ab          # noqa: E402
from eval import benchmark as bench        # noqa: E402

DEVICE = "cpu"
SPEC_PATH = "cache/bench_person_val.json"
CKPT_PATH = "checkpoints/stage_f96/last.pt"
LAMA_PATH = "models/big-lama.pt"
N_CASES = 2

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SPEC_PATH) and os.path.exists(CKPT_PATH)
         and os.path.exists(LAMA_PATH)),
    reason=f"{SPEC_PATH} / {CKPT_PATH} / {LAMA_PATH} のいずれかが無い "
           "(このリポジトリの標準配置が前提)")


# --- 1. --build がケース数分の PNG / key.json / votes.txt を作る -----------

def test_build_creates_expected_outputs(tmp_path):
    out = str(tmp_path / "ab")
    ab.build(SPEC_PATH, CKPT_PATH, N_CASES, out, DEVICE, seed=0, thumb=64)

    pngs = sorted(glob.glob(os.path.join(out, "*.png")))
    assert len(pngs) == N_CASES

    key_path = os.path.join(out, "key.json")
    votes_path = os.path.join(out, "votes.txt")
    assert os.path.exists(key_path)
    assert os.path.exists(votes_path)

    key = json.load(open(key_path))
    assert len(key["cases"]) == N_CASES
    assert all(v in ("A", "B") for v in key["cases"].values())

    votes_text = open(votes_path, encoding="utf-8").read()
    assert votes_text.startswith("#")   # 記入方法のコメントがある
    non_comment = [ln for ln in votes_text.splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
    assert len(non_comment) == N_CASES
    for name in key["cases"]:
        assert any(ln.startswith(name) for ln in non_comment)


# --- 2. 左右の無作為化が偏っていない ----------------------------------------

def test_assign_sides_not_biased():
    """200ケース分の割り当てを seed を変えて生成し、ours が A になる割合が
    0.4〜0.6 に収まることを確認する (常に同じ側に偏ると判定が汚染されるため)。
    """
    case_names = [f"case_{i}" for i in range(200)]
    counts = []
    for seed in range(5):
        sides = ab.assign_sides(case_names, seed)
        ours_a = sum(1 for v in sides.values() if v)
        counts.append(ours_a / len(case_names))
    mean_ratio = float(np.mean(counts))
    print(f"\n[assign_sides] seed毎の A 割合: {counts}  平均={mean_ratio:.3f}")
    for ratio in counts:
        assert 0.4 <= ratio <= 0.6


# --- 3. 同じ seed なら同じ割り当てになる (再現性) ---------------------------

def test_assign_sides_reproducible():
    case_names = [f"case_{i}" for i in range(50)]
    a1 = ab.assign_sides(case_names, seed=42)
    a2 = ab.assign_sides(case_names, seed=42)
    assert a1 == a2

    a3 = ab.assign_sides(case_names, seed=43)
    assert a3 != a1   # 異なる seed では (ほぼ確実に) 異なる割り当てになる


# --- 4. --tally が正しく集計する --------------------------------------------

def test_tally_known_win_counts():
    """人工的な votes/key で既知の勝敗数・p 値になることを確認する。

    10戦全勝なら両側 p = 2 * (1/2)^10 = 2/1024 ≈ 0.001953 になるはず。
    """
    key_map = {f"case_{i}": ("A" if i % 2 == 0 else "B") for i in range(10)}
    # 全ケースで「我々の側」を判定として書く = 我々の 10 戦 10 勝
    votes = {name: side for name, side in key_map.items()}

    result = ab.tally_votes(votes, key_map)
    assert result["wins"] == 10
    assert result["losses"] == 0
    assert result["ties"] == 0
    assert result["n"] == 10
    assert result["winrate"] == pytest.approx(1.0)

    expected_p = 2 * (0.5 ** 10)
    print(f"\n[tally] p={result['pvalue']:.6f}  既知値={expected_p:.6f}")
    assert result["pvalue"] == pytest.approx(expected_p, rel=1e-6)

    # 信頼区間は 0 と 1 の間で、勝率 1.0 を含む
    assert 0.0 <= result["ci_lo"] <= result["ci_hi"] <= 1.0


def test_tally_mixed_and_ties():
    """勝ち・負け・同等が混在するケース。同等は勝率計算・検定から除外される。"""
    key_map = {"c0": "A", "c1": "A", "c2": "B", "c3": "B", "c4": "A"}
    votes = {
        "c0": "A",     # 我々の判定と一致 -> 勝ち
        "c1": "B",     # 不一致 -> 負け
        "c2": "B",     # 一致 -> 勝ち
        "c3": "同等",  # 除外
        "c4": "A",     # 一致 -> 勝ち
    }
    result = ab.tally_votes(votes, key_map)
    assert result["wins"] == 3
    assert result["losses"] == 1
    assert result["ties"] == 1
    assert result["n"] == 4   # 同等を除いた件数


def test_binomial_test_matches_known_value():
    """自前の二項検定が既知の値と一致すること (scipy 非依存の実装の検証)。"""
    p10 = ab.binomial_test_two_sided(10, 10)
    assert p10 == pytest.approx(2 * (0.5 ** 10), rel=1e-6)

    # 5戦5勝 (公平なコインでも起きやすい): p = 2 * (1/2)^5 = 1/16 = 0.0625
    p5 = ab.binomial_test_two_sided(5, 5)
    assert p5 == pytest.approx(2 * (0.5 ** 5), rel=1e-6)

    # 5戦のうち3勝2敗は五分五分に近いので有意ではない (p は高い)
    p32 = ab.binomial_test_two_sided(3, 5)
    assert p32 > 0.5


def test_parse_votes_file(tmp_path):
    path = tmp_path / "votes.txt"
    path.write_text(
        "# コメント行\n"
        "\n"
        "case_a A\n"
        "case_b 同等\n"
        "case_c B\n",
        encoding="utf-8")
    votes = ab.parse_votes_file(str(path))
    assert votes == {"case_a": "A", "case_b": "同等", "case_c": "B"}


def test_tally_end_to_end(tmp_path):
    """`build` の出力を模した key.json / votes.txt を作り、`tally` を通す。"""
    out = tmp_path / "ab"
    out.mkdir()
    key_map = {f"case_{i}": ("A" if i % 3 == 0 else "B") for i in range(9)}
    json.dump(dict(spec="x", seed=0, cases=key_map), open(out / "key.json", "w"))

    lines = [ab.VOTES_HEADER]
    for name, side in key_map.items():
        lines.append(f"{name} {side}\n")   # 全部我々の勝ち
    (out / "votes.txt").write_text("".join(lines), encoding="utf-8")

    result = ab.tally(str(out))
    assert result["wins"] == 9
    assert result["losses"] == 0
    assert result["n"] == 9


# --- 5. 対比較統計 (eval.benchmark.paired_comparison) ----------------------

def test_paired_comparison_known_input():
    """全ケースで我々が +1.0dB のとき、中央値 +1.0、信頼区間が 0 を跨がない。"""
    ours = [dict(psnr=11.0 + 0.0) for _ in range(20)]
    base = [dict(psnr=10.0) for _ in range(20)]
    stat = bench.paired_comparison(ours, base, "psnr")

    assert stat["n"] == 20
    assert stat["median"] == pytest.approx(1.0)
    assert stat["ci_lo"] <= 1.0 <= stat["ci_hi"]
    assert stat["ci_lo"] > 0.0   # 0 を跨がない
    assert stat["pvalue"] < 0.01


def test_paired_comparison_no_difference():
    """差が無ければ (符号がランダム)、信頼区間は 0 付近をまたぐはず。"""
    rng = np.random.default_rng(0)
    diffs = rng.normal(0, 1.0, size=200)
    ours = [dict(psnr=10.0 + d) for d in diffs]
    base = [dict(psnr=10.0) for _ in diffs]
    stat = bench.paired_comparison(ours, base, "psnr")

    assert stat["ci_lo"] < 0.0 < stat["ci_hi"]


def test_bootstrap_median_ci_and_sign_test():
    empty_lo, empty_hi = bench.bootstrap_median_ci([])
    assert np.isnan(empty_lo) and np.isnan(empty_hi)

    lo, hi = bench.bootstrap_median_ci([1.0] * 10, n_boot=500, seed=1)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)

    p10 = bench.sign_test_pvalue([1.0] * 10)
    assert p10 == pytest.approx(2 * (0.5 ** 10), rel=1e-6)
