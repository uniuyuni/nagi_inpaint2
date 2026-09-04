"""`eval/postprocess_eval.py` と `eval/ab_compare.py --mode postproc` のテスト。
device は "cpu" 固定。

**背景**: LaMa の出力に Stage D を後処理として掛けると、指標上は改善して見え
ても盲検では別の話になりうる (Stage F 単体への後処理適用で PSNR が
24.35 -> 23.05 と悪化した前例がある)。`eval/postprocess_eval.py` の docstring
に書いた通り、これは「Stage D が自前の粗い出力にだけ効き、LaMa のような既に
連続性の高い出力には効かない (むしろ壊す)」可能性があるためで、本テストは
それを検出できる実装になっているかを確認する。

**Stage D の契約についての判断**: `stage_d.harmonize()` は fill (マスク外にも
伸びている合成結果) を要求する。LaMa の生出力はマスク外にも LaMa 自身が
計算した値が入っている一方、`np.where(mask, fill, img)` で作った「表示用」の
版はマスク外が元画像と恒等的に一致する。後者を harmonize に渡すと境界の
食い違いが 0 になり調和膜が消える (`nagi_inpaint2/stage_d.py` の fill 引数の
docstring に明記、`tests/test_stage_d.py::test_poisson_needs_fill_outside_mask`
で既に固定されている)。`eval/postprocess_eval.py` はこれを避けて**生出力**を
渡す設計にしている。このテストではその判断を実測で再確認・固定する
(`test_contract_masked_input_kills_poisson_membrane`,
`test_lama_raw_output_differs_outside_mask`)。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import ab_compare as ab                # noqa: E402
from eval import postprocess_eval as pp           # noqa: E402
from nagi_inpaint2 import stage_d                  # noqa: E402

DEVICE = "cpu"
SPEC_PATH = "cache/bench_person_val.json"
CKPT_PATH = "checkpoints/stage_f96/last.pt"
LAMA_PATH = "models/big-lama.pt"
N_CASES = 2

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SPEC_PATH) and os.path.exists(LAMA_PATH)),
    reason=f"{SPEC_PATH} / {LAMA_PATH} のいずれかが無い "
           "(このリポジトリの標準配置が前提)")


# --- 1. postprocess_eval の主要関数が import でき、2ケースで動作する ---------

def test_run_and_summarize_smoke(capsys):
    acc = pp.run(SPEC_PATH, N_CASES, DEVICE, seed=0)

    expected_names = {"LaMa", *pp.POSTPROC_LABELS.values()}
    assert set(acc) == expected_names

    for name, rs in acc.items():
        assert 0 < len(rs) <= N_CASES, f"{name}: 空、またはケース数超過"
        for r in rs:
            assert "psnr" in r and "boundary" in r and "grain" in r and "psd" in r

    # 同じケースを同じ順で各構成に積んでいる (paired_comparison の前提)
    n = len(acc["LaMa"])
    for name, rs in acc.items():
        assert len(rs) == n, f"{name} のケース数が LaMa と揃っていない"

    pp.summarize(acc)
    out = capsys.readouterr().out
    assert "LaMa" in out
    assert "対比較" in out


def test_run_with_lpips_smoke():
    """--lpips 相当 (eval.perceptual.Perceptual) を通しても壊れないこと。"""
    acc = pp.run(SPEC_PATH, N_CASES, DEVICE, seed=0, use_lpips=True)
    for rs in acc.values():
        for r in rs:
            assert "lpips" in r
            assert np.isfinite(r["lpips"])


# --- 2. Stage D の契約の確認 ------------------------------------------------

def _textured(n=96, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n] / n
    base = 0.5 + 0.2 * np.sin(6 * yy) + 0.15 * np.cos(5 * xx)
    img = base[..., None] + rng.normal(0, 0.02, (n, n, 3))
    return img.astype(np.float32)


def _square_mask(n=96, lo=32, hi=64):
    m = np.zeros((n, n), dtype=bool)
    m[lo:hi, lo:hi] = True
    return m


def test_contract_masked_input_kills_poisson_membrane():
    """マスク外が元画像と完全一致する入力を harmonize に渡すと調和膜が消える
    ことを実測して固定する (`eval/postprocess_eval.py` が「生出力を渡す」判断
    をした根拠そのもの)。

    `raw_fill` は LaMa の生出力を模す (マスク外にも本物と異なる値が伸びている)。
    そのまま harmonize すれば調和膜が効くが、先に
    `np.where(mask, raw_fill, img)` でマスク外を元画像に潰してから渡すと、
    境界の食い違いが恒等的に 0 になり、出力はマスク内側で入力と一致する
    (= 調和膜が「消える」)。
    """
    img = _textured()
    m = _square_mask()
    raw_fill = img * 0.9 + 0.05   # 全面にわたり元画像と異なる (ぼかし+オフセット相当)

    out_raw = stage_d.harmonize(raw_fill, img, m, rng=np.random.default_rng(0),
                                do_color=False, do_poisson=True, do_grain=False)
    assert not np.allclose(out_raw[m], raw_fill[m]), \
        "生出力を渡した場合は調和膜が効くはず (改善以前に何かは起きるはず)"

    masked_fill = np.where(m[..., None], raw_fill, img)
    out_masked = stage_d.harmonize(masked_fill, img, m, rng=np.random.default_rng(0),
                                   do_color=False, do_poisson=True, do_grain=False)
    # 実際の挙動を assert で固定する: マスク外=元画像の入力では調和膜が消え、
    # 出力はマスク内側で入力 (masked_fill) とそのまま一致する。
    assert np.allclose(out_masked[m], masked_fill[m], atol=1e-6), \
        "マスク外を先に元画像へ戻すと調和膜が消える、という契約が崩れている"


def test_lama_raw_output_differs_outside_mask():
    """実際の LaMa (big-lama.pt) は恒等写像ではないので、生出力はマスク外でも
    元画像とわずかに異なることを確認する。これが「生出力を渡す」判断の
    実測上の裏付けになる (マスク外が元画像と厳密に一致するなら、生出力を
    渡しても masked 版と同じ = 調和膜が消えるのと変わらない)。
    """
    from eval.benchmark import load_shard_tiles
    from eval.lama_baseline import Lama
    from masks import sample_mask

    spec = json.load(open(SPEC_PATH))
    c = spec["cases"][0]
    tiles = {k: (im, sm, mt) for k, im, sm, mt
             in load_shard_tiles(os.path.join("shards", c["shard"]), stride=1)}
    img, satm, _ = tiles[c["key"]]
    mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                          kind=c["kind"])

    lama = Lama(device=DEVICE)
    fill = lama(img, mask)

    outside = ~stage_d.dilate(mask, 1)
    assert outside.sum() > 0
    diff = float(np.abs(fill[outside] - img[outside]).mean())
    print(f"\n[lama raw] マスク外の平均絶対差 = {diff:.6f}")
    assert diff > 1e-6, "LaMa の生出力がマスク外で元画像と厳密に一致している " \
                        "(この場合は masked 版を渡しても調和膜への影響は同じになる)"


# --- 3. --mode postproc の A/B 画像生成 -------------------------------------

def test_build_mode_postproc_creates_expected_outputs(tmp_path):
    out = str(tmp_path / "ab_pp")
    ab.build(SPEC_PATH, "", N_CASES, out, DEVICE, seed=0, thumb=64,
             mode="postproc", postproc="poisson")

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
    non_comment = [ln for ln in votes_text.splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
    assert len(non_comment) == N_CASES
    for name in key["cases"]:
        assert any(ln.startswith(name) for ln in non_comment)


@pytest.mark.parametrize("postproc", ["poisson", "poisson_color", "poisson_grain", "all"])
def test_build_mode_postproc_all_choices(tmp_path, postproc):
    """--postproc の4択すべてでクラッシュせず画像が作れること。"""
    out = str(tmp_path / f"ab_pp_{postproc}")
    ab.build(SPEC_PATH, "", 1, out, DEVICE, seed=0, thumb=48,
             mode="postproc", postproc=postproc)
    pngs = glob.glob(os.path.join(out, "*.png"))
    assert len(pngs) == 1


def test_build_mode_invalid_raises():
    with pytest.raises(ValueError):
        ab.build(SPEC_PATH, "", 1, "out/should_not_be_created", DEVICE, seed=0,
                 mode="bogus")


# --- 4. 既存の --build (LaMa 対 我々) の挙動が変わっていないこと ------------

@pytest.mark.skipif(not os.path.exists(CKPT_PATH),
                    reason=f"{CKPT_PATH} が無い (このリポジトリの標準配置が前提)")
def test_build_mode_ours_default_unchanged(tmp_path):
    """`--mode` 未指定 (既定 'ours') の挙動が従来どおりであることを確認する。

    `tests/test_ab.py::test_build_creates_expected_outputs` と同じ形の
    アサーションを、`mode`/`postproc` 引数を追加した後の `build()` に対して
    もう一度通す。
    """
    out = str(tmp_path / "ab_ours")
    ab.build(SPEC_PATH, CKPT_PATH, N_CASES, out, DEVICE, seed=0, thumb=64)

    pngs = sorted(glob.glob(os.path.join(out, "*.png")))
    assert len(pngs) == N_CASES

    key = json.load(open(os.path.join(out, "key.json")))
    assert len(key["cases"]) == N_CASES
    assert all(v in ("A", "B") for v in key["cases"].values())

    votes_text = open(os.path.join(out, "votes.txt"), encoding="utf-8").read()
    non_comment = [ln for ln in votes_text.splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
    assert len(non_comment) == N_CASES


@pytest.mark.skipif(not os.path.exists(CKPT_PATH),
                    reason=f"{CKPT_PATH} が無い (このリポジトリの標準配置が前提)")
def test_build_mode_ours_explicit_matches_default(tmp_path):
    """`mode="ours"` を明示しても既定と同じ結果 (同じ seed で同じ割り当て) になること。"""
    out_default = str(tmp_path / "ab_default")
    out_explicit = str(tmp_path / "ab_explicit")
    ab.build(SPEC_PATH, CKPT_PATH, N_CASES, out_default, DEVICE, seed=0, thumb=64)
    ab.build(SPEC_PATH, CKPT_PATH, N_CASES, out_explicit, DEVICE, seed=0, thumb=64,
             mode="ours")

    key_default = json.load(open(os.path.join(out_default, "key.json")))["cases"]
    key_explicit = json.load(open(os.path.join(out_explicit, "key.json")))["cases"]
    assert key_default == key_explicit
