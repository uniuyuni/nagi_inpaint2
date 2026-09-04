"""LaMa が実際に失敗する条件を、自前データ横断で探す (SPEC §1.3)。

**方針の転換**

これまでは「LaMa の弱点」を仮定して設計し、その仮定が3つとも外れた
(継ぎ目・粒状・解像度。§3 冒頭と §4.4 の訂正を参照)。
仮定をやめ、**12,000枚の自前データを横断して失敗例を実測で収集する**。

失敗が見つかればそれが本プロジェクトの存在理由になり、
Stage A/B/C をその条件へ最適化できる。見つからなければ目標を見直す。

出力:
  cache/lama_failures.jsonl  全サンプルの指標 + シーン情報
  out/lama_worst_*.png       指標が悪い順のコンタクトシート (元 / マスク / LaMa)

**指標だけで判断しない。** LaMa の失敗は構造の捏造・にじみ・繰り返しとして出るため、
PSNR には現れないことがある。必ずコンタクトシートを目視すること。

  pixi run python eval/lama_failures.py --n 400
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import zstandard as zstd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.lama_baseline import Lama                  # noqa: E402
from eval.metrics import report                      # noqa: E402
from masks import sample_mask                        # noqa: E402
from nagi_inpaint2.color import byte_unshuffle, dequantize   # noqa: E402

OUT = "cache/lama_failures.jsonl"


def p(*a):
    print(*a, flush=True)


def sample_tiles(shard_dir: str, n: int, seed: int = 0):
    """多数の shard を横断してタイルを集める。

    shard は画像順に詰まっているので、shard を散らして取ると
    シーンの多様性がそのまま確保される。
    """
    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".tar"))
    rng = np.random.default_rng(seed)
    pick = shards[::max(1, len(shards) // 40)][:40]
    per = max(1, n // len(pick))
    d = zstd.ZstdDecompressor()
    out = []
    for s in pick:
        with tarfile.open(os.path.join(shard_dir, s)) as tf:
            keys = sorted(m[:-5] for m in tf.getnames() if m.endswith(".json"))
            if not keys:
                continue
            for k in rng.choice(keys, size=min(per, len(keys)), replace=False):
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


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x / 1.2, 0, 1) ** (1 / 1.0) * 255).astype(np.uint8)


def contact_sheet(rows, path: str, thumb: int = 320):
    """(タイトル, 元, マスク合成, LaMa) の行を並べる。"""
    w = thumb * 3 + 16
    h = sum(thumb + 20 for _ in rows)
    sheet = Image.new("RGB", (w, h), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    y = 0
    for title, img, masked, out in rows:
        for i, a in enumerate((img, masked, out)):
            im = Image.fromarray(to_u8(a)).resize((thumb, thumb), Image.BILINEAR)
            sheet.paste(im, (i * (thumb + 8), y))
        y += thumb
        d.text((4, y + 4), title, fill=(235, 235, 235))
        y += 20
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheet.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--worst", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    tiles = sample_tiles(a.shards, a.n, a.seed)
    p(f"タイル {len(tiles)} 件 (shard 横断)\n")
    lama = Lama(device=a.device)
    rng = np.random.default_rng(a.seed)

    recs, keep = [], []
    t0 = time.perf_counter()
    for i, (img, satm, meta) in enumerate(tiles, 1):
        mask, mmeta = sample_mask(meta["tile"], rng)
        if mask.sum() < 256:
            continue
        fill = lama(img, mask)
        out = np.where(mask[..., None], fill, img)
        m = report(out, img, mask, gt=img)
        rec = dict(shoot=meta.get("shoot"), body=meta.get("body"),
                   iso=meta.get("iso"), role=meta.get("role"),
                   std=meta.get("std"), sat_frac=meta.get("sat_frac"),
                   mask_kind=mmeta["kind"], mask_area=mmeta["area"],
                   **{k: (None if not np.isfinite(v) else round(float(v), 5))
                      for k, v in m.items()})
        recs.append(rec)
        keep.append((rec, img, out, mask))
        if i % 50 == 0:
            el = time.perf_counter() - t0
            p(f"  {i}/{len(tiles)}  {el/i*1000:.0f} ms/枚")

    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    p(f"\n{len(recs):,} 件 -> {a.out}")

    ok = [r for r in recs if r.get("psnr") is not None]
    ps = np.array([r["psnr"] for r in ok])
    p(f"\nPSNR 分布: p5 {np.percentile(ps,5):.1f}  p25 {np.percentile(ps,25):.1f}  "
      f"p50 {np.percentile(ps,50):.1f}  p75 {np.percentile(ps,75):.1f}  "
      f"p95 {np.percentile(ps,95):.1f} dB")

    p("\n=== マスク種別ごとの PSNR 中央値 ===")
    by = collections.defaultdict(list)
    for r in ok:
        by[r["mask_kind"]].append(r["psnr"])
    for k, v in sorted(by.items(), key=lambda kv: np.median(kv[1])):
        p(f"  {k:<8} n={len(v):>4}  {np.median(v):6.2f} dB")

    p("\n=== マスク面積帯ごとの PSNR 中央値 ===")
    bins = [(0, .005), (.005, .015), (.015, .03), (.03, .06), (.06, 1)]
    for lo, hi in bins:
        v = [r["psnr"] for r in ok if lo <= r["mask_area"] < hi]
        if v:
            p(f"  {lo*100:4.1f}-{hi*100:4.1f}%  n={len(v):>4}  {np.median(v):6.2f} dB")

    p("\n=== PSNR が低いショット上位 (シーン依存の失敗) ===")
    sh = collections.defaultdict(list)
    for r in ok:
        sh[r["shoot"]].append(r["psnr"])
    worst = sorted(((np.median(v), k, len(v)) for k, v in sh.items() if len(v) >= 3))
    for med, k, n in worst[:10]:
        p(f"  {med:6.2f} dB  n={n:>3}  {k}")

    # 目視用: PSNR 最悪と boundary 最悪
    keep_ok = [k for k in keep if k[0].get("psnr") is not None]
    for tag, key in (("psnr", lambda k: k[0]["psnr"]),
                     ("boundary", lambda k: -abs((k[0].get("boundary") or 1) - 1))):
        sel = sorted(keep_ok, key=key)[:a.worst]
        rows = [(f"{r['shoot']}  {r['mask_kind']} {r['mask_area']*100:.1f}%  "
                 f"psnr {r['psnr']:.1f} boundary {r.get('boundary')}",
                 img, np.where(msk[..., None], 0.0, img), out)
                for r, img, out, msk in sel]
        path = f"out/lama_worst_{tag}.png"
        contact_sheet(rows, path)
        p(f"\n目視用: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
