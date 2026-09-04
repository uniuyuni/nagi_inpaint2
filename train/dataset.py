"""shard から学習バッチを供給する (SPEC §6.4, §7)。

`data/extract.py` が書いた tar shard を読み、知覚空間タイルとマスクを返す。

**tar を順に舐めるだけにする理由**: tar はランダムアクセスに向かない。
webdataset と同じく「shard の順序をシャッフル + shard 内をバッファでシャッフル」
という二段で十分な混合が得られる。P2 の教訓（MPS ではランダムアクセスが高い）と同じで、
**連続読みに寄せるほうが速い**。

返すもの (すべて知覚空間 float32):

    image : (3, H, W)   元タイル。学習の GT
    mask  : (1, H, W)   埋める領域。1 = 穴
    valid : (1, H, W)   使ってよい画素 (飽和マップの否定)。
                        highlight_mode=5 の再構成画素は実測値ではないので
                        損失からも参照候補からも外す (§6.0)
"""
from __future__ import annotations

import io
import json
import os
import random
import tarfile

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import zstandard as zstd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from masks import sample_mask                                   # noqa: E402
from nagi_inpaint2.color import byte_unshuffle, dequantize      # noqa: E402


def list_shards(shard_dir: str) -> list[str]:
    return sorted(os.path.join(shard_dir, f)
                  for f in os.listdir(shard_dir) if f.endswith(".tar"))


def split_shards(shards: list[str], val_frac: float = 0.02, seed: int = 0):
    """shard 単位で train/val を分ける。**無作為に選ぶこと。**

    タイル単位で分けると同一画像由来のタイルが両側に入り、評価が甘くなるので
    shard 単位で分けるのは正しい。しかし**先頭 N 本を取ってはならない**。
    shard は (機種, ショット) 順に詰まっているため、先頭5本は
    すべて「2019-02-01 Akasaka Palace」の単一ショットになっていた。
    実測でその保留セットは学習側より内容が 1.47倍難しく、
    「過学習」と「ドメイン差」を切り分けられなくなっていた。
    """
    rng = random.Random(seed)
    idx = list(range(len(shards)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(shards) * val_frac)))
    va = {idx[i] for i in range(n_val)}
    return ([s for i, s in enumerate(shards) if i not in va],
            [s for i, s in enumerate(shards) if i in va])


class ShardTiles(IterableDataset):
    """tar shard を舐めてタイルを返す IterableDataset。

    shuffle_buf : shard 内シャッフル用のリングバッファ段数
    mask_kind   : None なら data/masks.py の重み付き分布からサンプルする
    """

    def __init__(self, shards: list[str], shuffle_buf: int = 512,
                 mask_kind: str | None = None, seed: int = 0,
                 infinite: bool = True):
        self.shards = list(shards)
        self.shuffle_buf = shuffle_buf
        self.mask_kind = mask_kind
        self.seed = seed
        self.infinite = infinite

    def _my_shards(self):
        info = get_worker_info()
        if info is None:
            return list(self.shards)
        return self.shards[info.id::info.num_workers]

    def _iter_tiles(self, path: str, dctx):
        """1 shard を順に舐める。tar は逐次読みが最速。"""
        with tarfile.open(path) as tf:
            group: dict[str, dict] = {}
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name
                key, _, ext = name.partition(".")
                g = group.setdefault(key, {})
                g[ext] = tf.extractfile(member).read()
                if len(g) < 3:
                    continue
                group.pop(key)
                try:
                    meta = json.loads(g["json"])
                    img = byte_unshuffle(dctx.decompress(g["img.u16.zst"]),
                                         tuple(meta["shape"]))
                    sat = np.load(io.BytesIO(dctx.decompress(g["sat.npy.zst"])))
                except Exception:
                    continue
                yield img, sat, meta

    def __iter__(self):
        shards = self._my_shards()
        info = get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))
        nrng = np.random.default_rng(self.seed + 1 + (info.id if info else 0))
        dctx = zstd.ZstdDecompressor()

        while True:
            rng.shuffle(shards)
            buf = []
            for path in shards:
                for img, sat, meta in self._iter_tiles(path, dctx):
                    buf.append((img, sat, meta))
                    if len(buf) < self.shuffle_buf:
                        continue
                    j = rng.randrange(len(buf))
                    buf[j], buf[-1] = buf[-1], buf[j]
                    yield self._make(buf.pop(), nrng)
            rng.shuffle(buf)
            for item in buf:
                yield self._make(item, nrng)
            if not self.infinite:
                return

    def _make(self, item, nrng):
        img_u16, sat, meta = item
        tile = meta["tile"]
        p = dequantize(img_u16, meta["q_scale"], meta["q_offset"])
        satm = np.unpackbits(sat)[:tile * tile].reshape(tile, tile).astype(bool)
        mask, mmeta = sample_mask(tile, nrng, kind=self.mask_kind)
        return dict(
            image=torch.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1))),
            mask=torch.from_numpy(mask[None].astype(np.float32)),
            valid=torch.from_numpy((~satm)[None].astype(np.float32)),
            mask_kind=mmeta["kind"],
            iso=int(meta.get("iso") or 0),
            role=meta.get("role", ""),
        )


def make_loader(shards, batch_size=4, workers=2, **kw):
    ds = ShardTiles(shards, **kw)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=workers,
        pin_memory=False, persistent_workers=workers > 0)


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()

    shards = list_shards(a.shards)
    tr, va = split_shards(shards)
    print(f"shard {len(shards)} 本 (train {len(tr)} / val {len(va)})")

    dl = make_loader(tr, batch_size=a.batch, workers=a.workers)
    it = iter(dl)
    next(it)                                    # ウォームアップ
    t0 = time.perf_counter()
    kinds, areas = {}, []
    for _ in range(a.steps):
        b = next(it)
        for k in b["mask_kind"]:
            kinds[k] = kinds.get(k, 0) + 1
        areas.append(float(b["mask"].mean()))
    dt = (time.perf_counter() - t0) / a.steps
    n = a.batch * a.steps
    print(f"\n{dt*1000:.0f} ms/batch  ({dt/a.batch*1000:.0f} ms/タイル, "
          f"{a.batch/dt:.0f} タイル/秒)")
    print(f"image {tuple(b['image'].shape)} {b['image'].dtype}  "
          f"範囲 [{b['image'].min():.3f}, {b['image'].max():.3f}]")
    print(f"マスク面積 平均 {np.mean(areas)*100:.2f}%")
    print("マスク種別: " + "  ".join(f"{k} {v/n*100:.0f}%" for k, v in sorted(kinds.items())))
