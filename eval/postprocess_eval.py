"""LaMa の出力に Stage D を後処理として掛け、素の LaMa と比較する (SPEC §1.3, §4.4, §9)。

**方針転換の背景**

このプロジェクトは「LaMa より良い補完を作る」ことを目指したが、人物サイズの穴
では盲検A/Bで **0勝16敗4分 (p=0.00003)** と全敗した。穴の内部を埋める能力では
勝てていない。

**しかし継ぎ目の処理では LaMa を上回っている**: Stage D (ポアソン調和) は
boundary を **2.215 → 0.820** に改善し、これは **LaMa 単体の 1.155 より良い**。

**効く理屈**: ならば **LaMa の出力に Stage D を掛ければ、LaMa 単体より良くなる
はず**。穴を埋める部分は外部 (LaMa) に委ね、こちらは「どんな補完でも品質を
上げる後処理層」として価値を出す。

**効かない条件**: Stage D が「自前の粗い出力にだけ効く」もので、LaMa の出力
(既に連続性が高い) には効かない、あるいは**むしろ壊す**可能性がある。実際、
過去に Stage F 単体の出力に Stage D を掛けると PSNR が **24.35 → 23.05** と
悪化した例がある。そのため本スクリプトは**素の LaMa との対比較 (中央値・
95%信頼区間・符号検定 p 値)** を必ず出す。数値だけでなく `eval/ab_compare.py
--mode postproc` で盲検A/B画像も作れるようにし、指標が裏切る可能性
(`eval/ab_compare.py` docstring 参照) に備える。

**Stage D の契約についての判断 (最重要)**

`nagi_inpaint2/stage_d.py` の `harmonize()` の docstring (fill 引数の説明) に
明記されている契約はこうである:

    fill : Stage B/C の合成結果。**マスクの外側にも最低 ANNULUS 幅まで伸びて
           いること。** 勾配ドメイン合成は「境界での元画像との食い違い」を
           境界条件に使うため、マスク外を先に元画像へ戻した fill を渡すと
           食い違いが恒等的に 0 になり、調和膜が消えて**何も起きない**。

LaMa の呼び出し (`eval/lama_baseline.py` の `Lama.__call__`) は
`self.m(x, m)` の**全画素の出力**をそのまま返す。マスク外を元画像に戻す
`np.where(mask, fill, img)` はその**後**に呼び出し側 (`eval/lama_baseline.py`
の `main()` や `eval/benchmark.py` の `add()`) が行っている。つまりこの
リポジトリでは「LaMa の生出力 (マスク外も LaMa 自身が計算した値)」と
「表示・最終出力用にマスク外を元画像へ戻した版」は明確に別物として扱われ
ており、Stage D には**常に生出力を渡す**のが既存の慣習である
(`eval/lama_baseline.py` L103-109、`eval/benchmark.py` L338 のコメント
「マスク外にも生出力を残したまま harmonize に渡す契約 (§4.4)」参照)。

本スクリプトもこの慣習に従い、**LaMa の生出力をそのまま `stage_d.harmonize`
に渡す**。マスクで合成した版 (`np.where(mask, fill, img)`) を渡すと、上記の
契約により調和膜が消えて Stage D が何もしなくなり、「LaMa+Stage D」が実質
「LaMa 単体」と同じ結果になってしまう (これは `tests/test_postproc.py` の
`test_contract_masked_input_kills_poisson_membrane` で実測して固定してある)。

なお LaMa は完全な恒等写像ではないため、生出力はマスク外でも元画像とわずかに
異なる (`tests/test_postproc.py` の `test_lama_raw_output_differs_outside_mask`
で実測)。したがって境界条件は非自明であり、調和膜が「効く余地はある」。
それが実際に品質を改善するか悪化させるかは、まさにこのスクリプトが測る対象
であり、ここでは判断しない。

  pixi run python eval/postprocess_eval.py --spec cache/bench_person_val.json \
      --cases 60 --device mps
  pixi run python eval/postprocess_eval.py --cases 60 --device mps --lpips
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.benchmark import load_shard_tiles, paired_comparison   # noqa: E402
from eval.lama_baseline import Lama, MODEL as LAMA_MODEL          # noqa: E402
from eval.metrics import report                                   # noqa: E402
from masks import sample_mask                                     # noqa: E402
from nagi_inpaint2 import stage_d                                  # noqa: E402

# 後処理構成: (do_color, do_poisson, do_grain)。
# キーは `eval/ab_compare.py --postproc` の選択肢と共有する
# (二重実装を避けるため、ab_compare.py 側はこの dict を import して使う)。
POSTPROC_CONFIGS: dict[str, dict[str, bool]] = {
    "poisson":       dict(do_color=False, do_poisson=True, do_grain=False),
    "poisson_color": dict(do_color=True, do_poisson=True, do_grain=False),
    "poisson_grain": dict(do_color=False, do_poisson=True, do_grain=True),
    "all":           dict(do_color=True, do_poisson=True, do_grain=True),
}

# 表示・レポート用の日本語ラベル。POSTPROC_CONFIGS と同じキー集合に対応する。
POSTPROC_LABELS: dict[str, str] = {
    "poisson":       "LaMa+ポアソン",
    "poisson_color": "LaMa+ポアソン+色",
    "poisson_grain": "LaMa+ポアソン+粒状",
    "all":           "LaMa+全部",
}


def p(*a):
    print(*a, flush=True)


def run(spec_path: str, cases: int, device: str, seed: int = 0,
        shard_dir: str = "shards", use_lpips: bool = False,
        model_path: str = LAMA_MODEL) -> dict[str, list[dict]]:
    """spec の先頭 `cases` 件で「LaMa 素」と「LaMa+各後処理」を評価する。

    戻り値: `{"LaMa": [...], "LaMa+ポアソン": [...], ...}`。各要素は
    `eval.metrics.report()` の dict (`use_lpips=True` なら `"lpips"` キーも
    足す)。**同じケースを同じ順で各構成に積んでいる**ので、返り値をそのまま
    `eval.benchmark.paired_comparison(acc[name], acc["LaMa"], key)` に渡せる
    (`paired_comparison` はインデックス対応を前提にしている)。

    契約についての判断はモジュール docstring 冒頭を参照。ここでは
    `lama(img, mask)` の**生出力**を `stage_d.harmonize` に渡す。
    """
    spec = json.load(open(spec_path))
    sel = spec["cases"][:cases]
    p(f"postprocess_eval: {len(sel)} ケース  device={device}")

    lama = Lama(model_path, device)
    lpips = None
    if use_lpips:
        from eval.perceptual import Perceptual
        lpips = Perceptual(device=device)

    names = ["LaMa"] + [POSTPROC_LABELS[k] for k in POSTPROC_CONFIGS]
    acc: dict[str, list[dict]] = {n: [] for n in names}

    by_shard: dict[str, list] = {}
    for c in sel:
        by_shard.setdefault(c["shard"], []).append(c)

    for shard, cs in by_shard.items():
        tiles = {k: (im, sm, mt) for k, im, sm, mt
                 in load_shard_tiles(os.path.join(shard_dir, shard), stride=1)}
        for c in cs:
            if c["key"] not in tiles:
                continue
            img, satm, _ = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            if mask.sum() == 0:
                continue

            # **契約判断: 生出力をそのまま harmonize に渡す** (モジュール
            # docstring 冒頭参照)。マスクで合成した版を渡すと調和膜が消える。
            fill = lama(img, mask)

            out_lama = np.where(mask[..., None], fill, img)
            r_lama = report(out_lama, img, mask, gt=img)
            if lpips is not None:
                r_lama["lpips"] = lpips(out_lama, img, mask=mask)
            acc["LaMa"].append(r_lama)

            for key, kwargs in POSTPROC_CONFIGS.items():
                out = stage_d.harmonize(fill, img, mask, valid=~satm,
                                        rng=np.random.default_rng(seed), **kwargs)
                r = report(out, img, mask, gt=img)
                if lpips is not None:
                    r["lpips"] = lpips(out, img, mask=mask)
                acc[POSTPROC_LABELS[key]].append(r)

    return acc


def summarize(acc: dict[str, list[dict]], use_lpips: bool = False) -> None:
    """指標の表と、各構成の「素の LaMa」対比較 (中央値・95%CI・符号検定 p値) を表示する。

    ブートストラップ信頼区間と符号検定は `eval.benchmark` の
    `bootstrap_median_ci` / `sign_test_pvalue` を内部で使う `paired_comparison`
    をそのまま呼ぶ (再実装しない)。
    """
    def med(rs, k):
        v = [r[k] for r in rs if np.isfinite(r.get(k, np.nan))]
        return float(np.median(v)) if v else float("nan")

    lpips_hdr = f"{'lpips':>8}" if use_lpips else ""
    p(f"\n{'構成':<20}{'boundary':>10}{'grain':>8}{'psd':>9}{lpips_hdr}"
      f"{'psnr':>8}{'spectral':>10}")
    for name, rs in acc.items():
        if not rs:
            continue
        lpips_col = f"{med(rs,'lpips'):>8.4f}" if use_lpips else ""
        p(f"{name:<20}{med(rs,'boundary'):>10.3f}{med(rs,'grain'):>8.3f}"
          f"{med(rs,'psd'):>9.4f}{lpips_col}{med(rs,'psnr'):>8.2f}"
          f"{med(rs,'spectral'):>10.4f}")

    base = acc.get("LaMa")
    if not base:
        return
    keys = ["psnr", "boundary", "grain", "psd", "spectral"]
    if use_lpips:
        keys.append("lpips")
    p("\n素の LaMa との対比較 (差 = 構成 - LaMa。有意かどうかは 95%CI が "
      "0 を跨がないかで見る):")
    for name, rs in acc.items():
        if name == "LaMa" or not rs:
            continue
        p(f"  [{name}]")
        for k in keys:
            stat = paired_comparison(rs, base, k)
            sig = "" if stat["ci_lo"] <= 0.0 <= stat["ci_hi"] else " *"
            p(f"    {k:<9} 差中央値 {stat['median']:+8.4f}  "
              f"95%CI [{stat['ci_lo']:+.4f}, {stat['ci_hi']:+.4f}]  "
              f"p={stat['pvalue']:.4f}  n={stat['n']}{sig}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="cache/bench_person_val.json")
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--model", default=LAMA_MODEL)
    ap.add_argument("--lpips", action="store_true",
                    help="eval/perceptual.py の Perceptual (LPIPS 相当) も測る")
    a = ap.parse_args()

    acc = run(a.spec, a.cases, a.device, seed=a.seed, shard_dir=a.shards,
              use_lpips=a.lpips, model_path=a.model)
    summarize(acc, use_lpips=a.lpips)
    return 0


if __name__ == "__main__":
    sys.exit(main())
