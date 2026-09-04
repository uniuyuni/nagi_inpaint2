"""LaMa をベースラインとして測る (SPEC §1.3, §9)。

**なぜ必要か**: Stage D の効果は「人工的に劣化させた fill を何dB改善したか」でしか
測れていなかった。実在手法に対する位置を知らないまま Stage B/C を作ると、
2週間かけた実装が良いのか悪いのか判定できない。LaMa は物差しである。

**同時に、Day 10-13 の成果物でもある** (§8):
「LaMa + 我々の Stage D」は学習ゼロで今日成立する改善であり、
Stage B/C が完成するまでの実用解にもなる。

モデル: big-lama の TorchScript 版 (Apache-2.0)。
LaMa 本体のコードベース (hydra / pytorch-lightning) は不要。
    https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
    sha256 7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c

**色空間の扱い**: LaMa は 8bit sRGB 前提で、我々は知覚空間 ProPhoto。
インペインティング能力そのものを比較したいので、色変換による損失を持ち込まず、
知覚空間の値をタイルごとに [0,1] へ線形正規化して渡し、出力を逆スケールで戻す。
知覚空間は既にガンマ状なので LaMa の学習分布と統計的に近い。
(ProPhoto -> sRGB 8bit の往復損失は LaMa 側パイプラインの制約であって、
 §1.3 で我々が主張する優位の一部。ここでは分離して測る。)

  pixi run python eval/lama_baseline.py --n 24
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.metrics import report                          # noqa: E402
from eval.validate_stage_d import load_tiles             # noqa: E402
from masks import sample_mask                            # noqa: E402
from nagi_inpaint2 import stage_d                        # noqa: E402


from nagi_inpaint2.lama import MODEL, NORM_PCT, Lama  # noqa: E402,F401



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    shards = sorted(f for f in os.listdir(a.shards) if f.endswith(".tar"))
    tiles = load_tiles(os.path.join(a.shards, shards[0]), a.n)
    lama = Lama(a.model, a.device)
    print(f"タイル {len(tiles)} 件  device={a.device}\n")

    rng = np.random.default_rng(a.seed)
    variants = {"LaMa 単体": [], "LaMa + Stage D": []}
    t_lama = t_d = 0.0
    n = 0

    for img, satm, meta in tiles:
        mask, _ = sample_mask(meta["tile"], rng)
        if mask.sum() < 256:
            continue
        n += 1

        t0 = time.perf_counter()
        fill = lama(img, mask)
        t_lama += time.perf_counter() - t0

        # LaMa 単体: マスク外は元画像に戻す (通常の使い方)
        out_lama = np.where(mask[..., None], fill, img)
        variants["LaMa 単体"].append(report(out_lama, img, mask, gt=img))

        # Stage D は fill がマスク外にも伸びていることを要求する (§4.4)。
        # LaMa はマスク外もそのまま返すので、生の出力をそのまま渡せる。
        t0 = time.perf_counter()
        out_d = stage_d.harmonize(fill, img, mask, valid=~satm,
                                  rng=np.random.default_rng(a.seed))
        t_d += time.perf_counter() - t0
        variants["LaMa + Stage D"].append(report(out_d, img, mask, gt=img))

    def med(rs, k):
        v = [r[k] for r in rs if np.isfinite(r.get(k, np.nan))]
        return float(np.median(v)) if v else float("nan")

    print(f"{'':<18}{'boundary':>10}{'grain':>8}{'psd':>9}{'psnr':>8}{'spectral':>10}")
    print(f"{'理想値':<18}{'1.00':>10}{'1.00':>8}{'0.00':>9}{'高い':>8}{'0.00':>10}")
    for name, rs in variants.items():
        if rs:
            print(f"{name:<18}{med(rs,'boundary'):>10.3f}{med(rs,'grain'):>8.3f}"
                  f"{med(rs,'psd'):>9.4f}{med(rs,'psnr'):>8.2f}"
                  f"{med(rs,'spectral'):>10.4f}")

    if n:
        print(f"\n速度: LaMa {t_lama/n*1000:.0f} ms/タイル  "
              f"Stage D {t_d/n*1000:.0f} ms/タイル")
    la, ld = variants["LaMa 単体"], variants["LaMa + Stage D"]
    if la and ld:
        print(f"\nStage D による改善:")
        for k, ideal in (("boundary", 1.0), ("grain", 1.0), ("psd", 0.0)):
            a0, a1 = med(la, k), med(ld, k)
            print(f"  {k:<9} {a0:7.3f} -> {a1:7.3f}  "
                  f"({'改善' if abs(a1-ideal) < abs(a0-ideal) else '悪化'})")
        print(f"  {'psnr':<9} {med(la,'psnr'):7.2f} -> {med(ld,'psnr'):7.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
