"""解像度軸での比較 — 本プロジェクトの前提が実在するかを検証する (SPEC §1.3)。

**なぜこれが決定的か**

512² タイル・小マスクの条件で測ったところ、LaMa は継ぎ目も粒状も既に良く、
Stage D は何も足さなかった (§4.4 の訂正)。SPEC が前提していた
「LaMa は継ぎ目を作る/そこだけツルツルになる」は誤りだった。

残る主張は解像度・色空間・速度の3つで、その中心が**解像度**である。
40MP を横切る電線に対し、LaMa には2つの手段しかない:

  a. 全体を縮小して適用し拡大で戻す -> 詳細が失われる
  b. タイル分割して適用           -> タイルごとに別の幻覚、境界に不整合

対して本設計は**フル解像度のまま、同じ画像の実画素をコピーする**。
ここで差がつかなければ、Stage B/C に2週間かける前に前提を見直す必要がある。

比較対象:
  lama_down  : 長辺 LAMA_SIZE に縮小 -> LaMa -> 拡大 -> 合成
  lama_tile  : 512² タイルに分割 -> タイルごとに LaMa -> 合成
  nonparam   : Stage B 非学習プロトタイプ (フル解像度で実画素コピー)
  nonparam+D : 上記 + Stage D

  pixi run python eval/resolution_test.py --n 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.lama_baseline import Lama                       # noqa: E402
from eval.metrics import report                           # noqa: E402
from masks import sample_mask                             # noqa: E402
from nagi_inpaint2 import stage_b_simple, stage_d         # noqa: E402



LAMA_SIZE = 1024      # 縮小適用時の長辺。LaMa の得意領域に置く
TILE = 512
CROP = 3072           # 評価に使う中央クロップ (フル 40MP は測定が重いため)


def p(*a):
    print(*a, flush=True)


def develop_crop(path: str, crop: int, cache_dir: str = "cache/devcrop") -> np.ndarray:
    """現像 + 中央クロップ。**別プロセスで実行する。**

    torch と libraw_enhanced は別々の libomp をリンクしており、同居させると
    OMP Error #15 になる。KMP_DUPLICATE_LIB_OK で握り潰すと結果が壊れ得るので
    プロセスを分ける。結果は .npy にキャッシュして再実行を速くする。
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{os.path.splitext(os.path.basename(path))[0]}_{crop}.npy"
    out = os.path.join(cache_dir, key)
    if not os.path.exists(out):
        env = dict(os.environ, PYTHONPATH="external/libraw_enhanced")
        subprocess.run([sys.executable, "eval/_develop_crop.py", path,
                        str(crop), out], check=True, env=env,
                       stdout=subprocess.DEVNULL)
    return np.load(out)


def wire_mask(h: int, w: int, rng) -> np.ndarray:
    """フル解像度を横切る電線マスク。実利用で最も多い除去対象 (§6.5)。

    タイル単位で生成した細線を並べるのではなく、画像全体を貫く1本にする。
    これが「LaMa が縮小かタイル分割を強いられる」状況そのもの。
    """
    from PIL import Image, ImageDraw
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    y0 = rng.uniform(h * 0.2, h * 0.8)
    y1 = rng.uniform(h * 0.2, h * 0.8)
    sag = rng.uniform(0.02, 0.08) * h
    pts = []
    for i in range(200):
        t = i / 199
        y = y0 * (1 - t) + y1 * t + sag * np.sin(np.pi * t)
        pts.append((t * w, y))
    d.line(pts, fill=255, width=int(rng.integers(5, 13)), joint="curve")
    return np.asarray(im) > 127


def lama_downscaled(lama: Lama, p_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """全体を縮小して LaMa をかけ、拡大して戻す。"""
    h, w = mask.shape
    s = LAMA_SIZE / max(h, w)
    nh, nw = int(round(h * s / 8) * 8), int(round(w * s / 8) * 8)
    x = torch.from_numpy(p_img.transpose(2, 0, 1)[None])
    m = torch.from_numpy(mask[None, None].astype(np.float32))
    xs = F.interpolate(x, (nh, nw), mode="area")
    ms = (F.interpolate(m, (nh, nw), mode="area") > 0.05).float()
    small = lama(xs[0].permute(1, 2, 0).numpy(), ms[0, 0].numpy())
    up = F.interpolate(
        torch.from_numpy(small.transpose(2, 0, 1)[None]), (h, w),
        mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    return up.astype(np.float32)


def lama_tiled(lama: Lama, p_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """512² タイルに分割してタイルごとに LaMa をかける。"""
    h, w = mask.shape
    out = p_img.copy()
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            ys, xs = slice(y, min(y + TILE, h)), slice(x, min(x + TILE, w))
            mt = mask[ys, xs]
            if not mt.any():
                continue
            th, tw = mt.shape
            if th < 64 or tw < 64:
                continue
            out[ys, xs] = lama(p_img[ys, xs], mt.astype(np.float32))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="cache/selection.jsonl")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.selection)]
    # X-T5 (40MP) の detail 役を使う
    recs = [r for r in recs if r["body"] == "X-T5" and r["role"] == "detail"]
    step = max(1, len(recs) // a.n)
    recs = recs[::step][:a.n]
    p(f"X-T5 {len(recs)} 枚 / 中央 {a.crop}² クロップで評価\n")

    lama = Lama(device=a.device)
    rng = np.random.default_rng(a.seed)
    acc = {k: [] for k in ("lama_down", "lama_tile", "nonparam", "nonparam+D")}
    times = {k: 0.0 for k in acc}

    for r in recs:
        c = a.crop
        img = develop_crop(r["path"], c)
        mask = wire_mask(c, c, rng)
        p(f"  {os.path.basename(r['path'])}  {c}x{c}  "
          f"マスク {mask.mean()*100:.2f}%")

        runs = {}
        t0 = time.perf_counter()
        runs["lama_down"] = lama_downscaled(lama, img, mask)
        times["lama_down"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        runs["lama_tile"] = lama_tiled(lama, img, mask)
        times["lama_tile"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        npm = stage_b_simple.inpaint_np(img, mask, device=a.device)
        times["nonparam"] += time.perf_counter() - t0
        runs["nonparam"] = npm

        t0 = time.perf_counter()
        runs["nonparam+D"] = stage_d.harmonize(
            npm, img, mask, rng=np.random.default_rng(a.seed),
            do_color=False, do_poisson=True, do_grain=False)
        times["nonparam+D"] += times["nonparam"] + (time.perf_counter() - t0)

        for k, v in runs.items():
            out = np.where(mask[..., None], v, img)
            acc[k].append(report(out, img, mask, gt=img))

    def med(rs, k):
        v = [x[k] for x in rs if np.isfinite(x.get(k, np.nan))]
        return float(np.median(v)) if v else float("nan")

    p(f"\n{'手法':<14}{'boundary':>10}{'grain':>8}{'psd':>9}"
      f"{'psnr':>8}{'spectral':>10}{'秒/枚':>9}")
    p(f"{'理想値':<14}{'1.00':>10}{'1.00':>8}{'0.00':>9}{'高い':>8}{'0.00':>10}{'':>9}")
    for k, rs in acc.items():
        if rs:
            p(f"{k:<14}{med(rs,'boundary'):>10.3f}{med(rs,'grain'):>8.3f}"
              f"{med(rs,'psd'):>9.4f}{med(rs,'psnr'):>8.2f}"
              f"{med(rs,'spectral'):>10.4f}{times[k]/len(recs):>9.1f}")

    p("\n判定: spectral が小さいほど高周波が保たれている。"
      "psnr は実画素コピーが有利に出やすい点に注意。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
