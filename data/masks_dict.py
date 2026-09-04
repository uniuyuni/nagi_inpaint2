"""実物体シルエットのマスク辞書を作る (SPEC §6.5)。

手続き生成 (`data/masks.py`) の `blob` は極座標で揺らした閉曲線にすぎず、
人物・看板・ゴミのような実物体の輪郭 (頭部+肩、四肢、突起、凹み) を模せない。
MobileSAM で自前画像から**実際のシルエットを数千枚抽出**し、
それを別の画像に貼ることで形状の多様性を自己教師的に確保する。

**MobileSAM は CPU で回す。** MPS は内部で float64 を使うため動かないが、
学習が GPU を占有している間に CPU で並行して回せるので、むしろ好都合。

**セグメンテーションは学習データの生成にしか使わない。** 製品には載らないので、
モデルのライセンス (Apache-2.0) が成果物に及ぶことはない。

抽出したマスクの選別基準:
  - 面積比 0.1% 〜 8% (§6.5 のスコープ)
  - 細長すぎない (充填率が低いものは葉の隙間などの背景断片)
  - タイル境界に張り付いていない (切れた物体は輪郭として不自然)

出力: cache/mask_dict.npz (packbits した 128x128 の二値マスク群)

  pixi run python data/masks_dict.py --tiles 800
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "external/MobileSAM")

from nagi_inpaint2.color import byte_unshuffle, dequantize      # noqa: E402

OUT = "cache/mask_dict.npz"
STORE = 128            # 辞書に格納する解像度 (使用時に拡縮する)
AREA_LO, AREA_HI = 0.001, 0.08
MIN_FILL = 0.35        # 外接矩形に対する充填率。低いものは背景の断片
MAX_BORDER = 0.35      # 境界に接する画素の割合の上限


def p(*a):
    print(*a, flush=True)


def iter_tiles(shard_dir: str, n: int, seed: int = 0):
    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".tar"))
    rng = np.random.default_rng(seed)
    pick = shards[::max(1, len(shards) // 40)][:40]
    per = max(1, n // len(pick))
    d = zstd.ZstdDecompressor()
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
                yield dequantize(img, meta["q_scale"], meta["q_offset"]), meta


def to_u8(p_img: np.ndarray) -> np.ndarray:
    """知覚空間 -> 8bit RGB。SAM は 8bit を前提とする。"""
    return (np.clip(p_img / 1.2, 0, 1) * 255).astype(np.uint8)


def keep(seg: np.ndarray) -> bool:
    h, w = seg.shape
    area = seg.mean()
    if not (AREA_LO <= area <= AREA_HI):
        return False
    ys, xs = np.nonzero(seg)
    if len(ys) < 64:
        return False
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if seg.sum() / max(bh * bw, 1) < MIN_FILL:       # 細長い / 散在
        return False
    border = (seg[0].sum() + seg[-1].sum() + seg[:, 0].sum() + seg[:, -1].sum())
    if border / max(2 * (h + w), 1) > MAX_BORDER:    # 切れた物体
        return False
    return True


def crop_store(seg: np.ndarray) -> np.ndarray:
    """外接矩形で切り出し STORE 角へ収める (アスペクト比は保つ)。"""
    from PIL import Image
    ys, xs = np.nonzero(seg)
    sub = seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray((sub * 255).astype(np.uint8))
    h, w = sub.shape
    s = STORE / max(h, w)
    im = im.resize((max(int(w * s), 1), max(int(h * s), 1)), Image.NEAREST)
    out = np.zeros((STORE, STORE), dtype=np.uint8)
    a = np.asarray(im) > 127
    y0, x0 = (STORE - a.shape[0]) // 2, (STORE - a.shape[1]) // 2
    out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--tiles", type=int, default=800)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--points", type=int, default=16)
    ap.add_argument("--device", default="cpu",
                    help="MPS は内部の float64 で動かない。CPU 固定でよい")
    ap.add_argument("--limit-per-tile", type=int, default=6)
    a = ap.parse_args()

    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_t"](
        checkpoint="external/MobileSAM/weights/mobile_sam.pt").eval().to(a.device)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points,
                                    min_mask_region_area=256)
    p(f"MobileSAM {sum(q.numel() for q in sam.parameters())/1e6:.1f}M  device={a.device}")

    masks, shoots = [], []
    t0 = time.perf_counter()
    n_tile = 0
    for p_img, meta in iter_tiles(a.shards, a.tiles):
        n_tile += 1
        try:
            segs = gen.generate(to_u8(p_img))
        except Exception as e:
            p(f"  skip {type(e).__name__}: {str(e)[:80]}")
            continue
        got = 0
        for s in sorted(segs, key=lambda d: -d["area"]):
            seg = s["segmentation"]
            if not keep(seg):
                continue
            masks.append(crop_store(seg))
            shoots.append(meta.get("shoot", ""))
            got += 1
            if got >= a.limit_per_tile:
                break
        if n_tile % 20 == 0:
            el = time.perf_counter() - t0
            p(f"  {n_tile}/{a.tiles} タイル  シルエット {len(masks):,}  "
              f"{el/n_tile:.1f}s/タイル  残り約 {el/n_tile*(a.tiles-n_tile)/60:.0f}分")

    if not masks:
        p("シルエットが取れなかった")
        return 1
    M = np.stack(masks)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, masks=np.packbits(M, axis=-1), size=STORE,
                        shoots=np.array(shoots))
    p(f"\n{len(masks):,} シルエット -> {a.out} "
      f"({os.path.getsize(a.out)/1e6:.1f} MB)")
    ar = M.reshape(len(M), -1).mean(1)
    p(f"格納時の充填率: p10 {np.percentile(ar,10):.3f}  p50 {np.percentile(ar,50):.3f}  "
      f"p90 {np.percentile(ar,90):.3f}")
    p(f"由来ショット数: {len(set(shoots))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
