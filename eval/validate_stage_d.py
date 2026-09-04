"""Stage D の検証 — 実タイルで「継ぎ目・色・粒状」が直るかを測る (SPEC §4.4, §9.1)。

抽出済み shard から実タイルを取り、素朴なインペインタが作りそうな欠陥を人工的に注入する:

    naive = ぼかし + 明るさオフセット + ノイズ除去

これは LaMa や拡散モデルがマスク内に出す典型的な失敗そのもの
(高周波が落ち、色がずれ、そこだけツルツルになる)。
そこへ Stage D を段階的に適用し、指標がどう動くかを見る。

指標 (eval/metrics.py):
    boundary  1.0 が「継ぎ目が見えない」
    grain     1.0 が「周囲と同じ粒状」。0 に近い = そこだけツルツル
    psd       0 が「粒の粗さまで一致」
    psnr      GT (元タイル) に対する再現度

  pixi run python eval/validate_stage_d.py --n 12
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile

import numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import report                       # noqa: E402
from nagi_inpaint2 import stage_d                      # noqa: E402
from nagi_inpaint2.color import byte_unshuffle, dequantize   # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
from masks import sample_mask                          # noqa: E402


def load_tiles(shard: str, n: int):
    """shard から (知覚空間タイル, 飽和マップ, meta) を n 件読む。"""
    d = zstd.ZstdDecompressor()
    out = []
    with tarfile.open(shard) as tf:
        keys = sorted(m[:-5] for m in tf.getnames() if m.endswith(".json"))
        step = max(1, len(keys) // n)
        for k in keys[::step][:n]:
            meta = json.loads(tf.extractfile(k + ".json").read())
            img = byte_unshuffle(
                d.decompress(tf.extractfile(k + ".img.u16.zst").read()),
                tuple(meta["shape"]))
            sat = np.load(io.BytesIO(
                d.decompress(tf.extractfile(k + ".sat.npy.zst").read())))
            satm = np.unpackbits(sat)[:meta["tile"] ** 2].reshape(
                meta["tile"], meta["tile"]).astype(bool)
            out.append((dequantize(img, meta["q_scale"], meta["q_offset"]),
                        satm, meta))
    return out


def _blur(x: np.ndarray, r: int = 4) -> np.ndarray:
    """箱平均を2回。ガウシアン相当のぼかし。"""
    k = np.ones(2 * r + 1) / (2 * r + 1)
    o = x.copy()
    for _ in range(2):
        for ax in (0, 1):
            o = np.apply_along_axis(
                lambda v: np.convolve(v, k, mode="same"), ax, o)
    return o


def _highpass(x: np.ndarray) -> np.ndarray:
    k = np.ones(3) / 3.0
    b = x.copy()
    for ax in (0, 1):
        b = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, b)
    return x - b


def make_fill(img: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    """欠陥を注入した fill を作る。

    **マスクの外側にも値を残す**こと。マスク外を元画像に戻してしまうと
    勾配ドメイン合成の境界条件が恒等的に 0 になり、調和膜が消えて何も起きない
    (これは実際に踏んだバグ)。

    kind="denoised" : Stage C の出力を模す。テクスチャは残しノイズだけ除く。
                      Stage D の担当範囲 (継ぎ目・色・粒状) を切り分けて測れる。
    kind="blurred"  : LaMa や拡散モデルの典型的な失敗。高周波ごと落ちる。
                      Stage D だけでは粒状比が 1.0 に届かない
                      (アニュラス側の高周波にはシーンのテクスチャが含まれるため)。
    """
    if kind == "denoised":
        fill = img - _highpass(img) * 0.9        # ノイズを消しテクスチャは残す
    else:
        fill = _blur(img, 4)
    fill = fill + rng.uniform(-0.05, 0.05)       # 明るさのずれ
    fill = fill * rng.uniform(0.92, 1.08)        # コントラストのずれ
    return fill.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=("denoised", "blurred"), default="denoised",
                    help="denoised=Stage C 出力を模す / blurred=LaMa 的な失敗")
    a = ap.parse_args()

    shards = sorted(f for f in os.listdir(a.shards) if f.endswith(".tar"))
    if not shards:
        print(f"shard がない: {a.shards}")
        return 1
    tiles = load_tiles(os.path.join(a.shards, shards[0]), a.n)
    print(f"タイル {len(tiles)} 件 ({shards[0]})  fill={a.kind}\n")

    rng = np.random.default_rng(a.seed)
    stages = [
        ("naive (何もしない)", dict(do_color=False, do_poisson=False, do_grain=False)),
        ("+ 色統計整合", dict(do_color=True, do_poisson=False, do_grain=False)),
        ("+ 勾配ドメイン", dict(do_color=True, do_poisson=True, do_grain=False)),
        ("+ 粒状注入 (全段)", dict(do_color=True, do_poisson=True, do_grain=True)),
    ]
    acc = {name: [] for name, _ in stages}

    for img, satm, meta in tiles:
        mask, _ = sample_mask(meta["tile"], rng)
        if mask.sum() < 256:
            continue
        naive = make_fill(img, rng, a.kind)
        for name, kw in stages:
            out = stage_d.harmonize(naive, img, mask, valid=~satm,
                                    rng=np.random.default_rng(a.seed), **kw)
            acc[name].append(report(out, img, mask, gt=img))

    def med(rs, k):
        v = [r[k] for r in rs if np.isfinite(r.get(k, np.nan))]
        return float(np.median(v)) if v else float("nan")

    print(f"{'段階':<22}{'boundary':>10}{'grain':>8}{'psd':>8}"
          f"{'psnr':>8}{'spectral':>10}")
    print(f"{'理想値':<22}{'1.00':>10}{'1.00':>8}{'0.00':>8}{'高い':>8}{'0.00':>10}")
    for name, _ in stages:
        rs = acc[name]
        if not rs:
            continue
        print(f"{name:<22}{med(rs,'boundary'):>10.3f}{med(rs,'grain'):>8.3f}"
              f"{med(rs,'psd'):>8.4f}{med(rs,'psnr'):>8.2f}"
              f"{med(rs,'spectral'):>10.4f}")

    print("\n判定:")
    b0, b3 = med(acc[stages[0][0]], "boundary"), med(acc[stages[3][0]], "boundary")
    g0, g3 = med(acc[stages[0][0]], "grain"), med(acc[stages[3][0]], "grain")
    print(f"  継ぎ目 {b0:.2f} -> {b3:.2f} "
          f"({'改善' if abs(b3-1) < abs(b0-1) else '悪化'})")
    print(f"  粒状   {g0:.2f} -> {g3:.2f} "
          f"({'改善' if abs(g3-1) < abs(g0-1) else '悪化'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
