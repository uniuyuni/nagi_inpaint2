"""Day 3-6 / 一括抽出パス — 選抜画像を現像し 512² タイル shard にする (SPEC §6.4)。

  RAW --(§6.0 の設定)--> リニア ProPhoto float32
      --(§3.3)--> 知覚空間 --> uint16 --> 512² タイル --> zstd shard

**設計上の要点**

- **現像設定は §6.0 に固定**。学習と推論で一致していないと Stage B の照合特徴と
  Stage D の粒状 PSD が両方壊れる。ここを変えてはならない。
- **`white` は画像全体で一度だけ決める** (§3.3)。タイルごとに再計算すると
  タイル境界に不連続が出る。
- **符号付きべき乗**。リニア ProPhoto には色域外の負値が正常に発生するため、
  `max(x,0)` でクリップすると彩度の高い被写体の境界で色が飛ぶ。
- **飽和マップを同梱**。`highlight_mode=5` が再構成した画素は実測値ではなく推定値で、
  現像後のリニア値が 1.0 を超えることで識別できる (実測で RAW 飽和率とよく対応)。
  Stage B の参照候補と Stage D のアニュラス推定の両方から除外するために持ち回る。
- **タイル数は画像ごとに可変**。X-T5 (40MP) は X-E3 (24MP) の約1.7倍のタイルを産むので、
  一律配分は X-T5 を過小評価する。使用可能面積に比例させ、上限で頭を打つ。

出力: shards/train-XXXX.tar (zstd 圧縮した .npy メンバ) と cache/extract_manifest.jsonl。
中断・再開可能 (manifest にある画像はスキップ)。

  pixi run python data/extract.py --limit 20      # 動作確認
  pixi run python data/extract.py

注意: /Volumes/PHOTOS 01 は読み出しのみ。書き込みは内蔵SSDのみ。
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

import libraw_enhanced as lre

SELECTION = "cache/selection.jsonl"
MANIFEST = "cache/extract_manifest.jsonl"
SHARD_DIR = "shards"

TILE = 512
# **単一の出典から取る。** ここに数値を書くと推論側 (nagi_inpaint2/api.py) と
# 二重管理になり、片方だけずれても誰も気づかない。あちらは torch を使うので
# このファイル (libraw_enhanced を import する) からは借りられない。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nagi_inpaint2.perceptual_space import GAMMA, WHITE_PCT  # noqa: E402
SAT_LEVEL = 1.0          # 現像後リニアでこれを超えたら再構成ハイライト

# 知覚空間 -> uint16。範囲 p in [-0.667, +4.79]、精度 8.3e-5 (約13.5bit)
Q_SCALE = 12000
Q_OFFSET = 8000

# タイル数は画素数に比例させる。X-T5 (40MP) は X-E3 (24MP) の約1.7倍のタイルを産むので、
# 一律配分では X-T5 を過小評価する。0.75/MP で X-E3 18 / X-T4 20 / X-T5 30 枚、
# 選抜 6,000枚の平均が約22枚 = 約13.2万タイル = 約138GB (§6.6 の 140GB 枠に収まる)。
TILES_PER_MP = 0.75
MAX_TILES_PER_IMAGE = 40
MIN_TILE_STD = 0.010     # 知覚空間での標準偏差。空や無地の壁を落とす
MAX_TILE_SAT = 0.25      # タイル内の再構成ハイライト率の上限
SHARD_TILES = 1000       # 1 shard あたりのタイル数 (約 1GB)

# 圧縮方式。16bit 写真データの下位ビットはノイズそのもので原理的に圧縮できないため、
# 素の zstd では 1.15x しか縮まない。上位/下位バイトを分離 (byte-shuffle) すると
# 上位バイト側が滑らかになり 1.50x まで改善する。実測:
#   zstd-3 1.371MB(1.15x, 2.2ms) / byte-shuffle+zstd-3 1.046MB(1.50x, 4.0ms)
#   delta+shuffle+zstd-9 0.987MB(1.59x, 17.7ms) / PNG-16 1.141MB(1.38x, 110ms)
# delta 版は 6% しか縮まないうえ符号化が4倍かかり、学習中の展開コストも増える。
# byte-shuffle の展開は reshape のみで済むので、そちらを採る。
ZSTD_LEVEL = 3

# §6.0 — ShadeWave/imageset.py:367-383 と同一。変更禁止。
# NOTE: X-Trans では AMaZE 指定でも実際には demosaic_xtrans_3pass が走る
#       (AMaZE は Bayer 専用)。ShadeWave と同じ指定を保つことが目的。
POSTPROCESS_KW = dict(
    output_color=lre.ColorSpace.ProPhotoRGB,
    demosaic_algorithm=lre.DemosaicAlgorithm.AMaZE,
    output_bps=32,
    use_camera_wb=True,
    gamma=(1.0, 1.0),
    half_size=False,
    highlight_mode=5,
    defringe=True,
    defringe_green=True,
    lateral_ca_correction=True,
    axial_ca_correction=True,
    use_gpu_acceleration=True,
    preprocess=False,
)


def p(*a):
    print(*a, flush=True)


def to_perceptual(img: np.ndarray) -> tuple[np.ndarray, float]:
    """リニア ProPhoto -> 知覚空間 (§3.3)。white は画像全体で一度だけ決める。"""
    from nagi_inpaint2.perceptual_space import to_perceptual as _tp
    return _tp(img)


def quantize(pimg: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(pimg * Q_SCALE) + Q_OFFSET, 0, 65535).astype(np.uint16)


def byte_shuffle(arr: np.ndarray) -> bytes:
    """uint16 配列の上位/下位バイトを分離する。読み出し側は逆順に組み直す。"""
    a = arr.view(np.uint8).reshape(-1, 2)
    return np.concatenate([a[:, 0], a[:, 1]]).tobytes()


def byte_unshuffle(buf: bytes, shape, dtype=np.uint16) -> np.ndarray:
    """byte_shuffle の逆。学習側のローダから使う。"""
    a = np.frombuffer(buf, dtype=np.uint8)
    half = len(a) // 2
    out = np.empty((half, 2), dtype=np.uint8)
    out[:, 0], out[:, 1] = a[:half], a[half:]
    return out.view(dtype).reshape(shape)


def pack_u16(arr: np.ndarray, cctx: zstd.ZstdCompressor) -> bytes:
    return cctx.compress(byte_shuffle(arr))


def npy_zstd(arr: np.ndarray, cctx: zstd.ZstdCompressor) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return cctx.compress(buf.getvalue())


class ShardWriter:
    def __init__(self, out_dir: str, tiles_per_shard: int):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.per = out_dir, tiles_per_shard
        existing = sorted(f for f in os.listdir(out_dir) if f.endswith(".tar"))
        self.idx = len(existing)
        self.tar = None
        self.n = 0
        self.total = 0
        self.cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)

    def _open(self):
        path = os.path.join(self.dir, f"train-{self.idx:04d}.tar")
        self.tar = tarfile.open(path, "w")
        self.n = 0

    def add(self, key: str, img: np.ndarray, sat: np.ndarray, meta: dict):
        if self.tar is None or self.n >= self.per:
            self.close()
            self._open()
        # 画素は byte-shuffle + zstd (形状は json 側に持たせる)。
        # 飽和マップは packbits 済みで小さいので素の .npy + zstd で足りる。
        for ext, payload in (
            ("img.u16.zst", pack_u16(img, self.cctx)),
            ("sat.npy.zst", npy_zstd(sat, self.cctx)),
            ("json", json.dumps(meta, ensure_ascii=False).encode()),
        ):
            info = tarfile.TarInfo(f"{key}.{ext}")
            info.size = len(payload)
            self.tar.addfile(info, io.BytesIO(payload))
        self.n += 1
        self.total += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None
            self.idx += 1


def pick_tiles(q: np.ndarray, satmask: np.ndarray, budget: int):
    """タイル候補を評価し、良い順に budget 枚返す。"""
    h, w = q.shape[:2]
    cands = []
    for y in range(0, h - TILE + 1, TILE):
        for x in range(0, w - TILE + 1, TILE):
            s = satmask[y:y + TILE, x:x + TILE]
            sat_frac = float(s.mean())
            if sat_frac > MAX_TILE_SAT:
                continue
            t = q[y:y + TILE, x:x + TILE]
            # uint16 のまま標準偏差を取り、知覚空間の単位に戻す
            std = float(t.std()) / Q_SCALE
            if std < MIN_TILE_STD:
                continue
            cands.append((std, y, x, sat_frac))
    cands.sort(reverse=True)
    return cands[:budget]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=SELECTION)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--shards", default=SHARD_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tiles", type=int, default=MAX_TILES_PER_IMAGE)
    a = ap.parse_args()

    if not os.path.exists(a.selection):
        p(f"選抜結果がない: {a.selection}  先に data/select_images.py を実行すること")
        return 1

    recs = [json.loads(l) for l in open(a.selection)]
    p(f"選抜 {len(recs):,} 枚")
    if a.limit:
        recs = recs[:a.limit]

    done = set()
    if os.path.exists(a.manifest):
        for line in open(a.manifest):
            try:
                done.add(json.loads(line)["path"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [r for r in recs if r["path"] not in done]
    if done:
        p(f"  処理済み {len(done):,} 枚をスキップ -> 残り {len(todo):,} 枚")
    if not todo:
        p("完了済み")
        return 0

    writer = ShardWriter(a.shards, SHARD_TILES)
    t0 = time.perf_counter()
    n_err = 0
    with open(a.manifest, "a") as fm:
        for i, r in enumerate(todo, 1):
            try:
                with lre.imread(r["path"]) as raw:
                    img = raw.postprocess(**POSTPROCESS_KW)
                satmask = (img > SAT_LEVEL).any(axis=2)
                pimg, white = to_perceptual(img)
                q = quantize(pimg)
                del img, pimg

                mp = q.shape[0] * q.shape[1] / 1e6
                budget = int(min(max(round(TILES_PER_MP * mp), 4), a.max_tiles))
                picked = pick_tiles(q, satmask, budget)
                stem = os.path.splitext(os.path.basename(r["path"]))[0]
                for n, (std, y, x, sat_frac) in enumerate(picked):
                    key = f"{r['body']}_{stem}_{n:03d}"
                    writer.add(
                        key,
                        q[y:y + TILE, x:x + TILE].copy(),
                        np.packbits(satmask[y:y + TILE, x:x + TILE]),
                        dict(path=r["path"], body=r["body"], shoot=r["shoot"],
                             iso=r["iso"], role=r["role"], white=white,
                             q_scale=Q_SCALE, q_offset=Q_OFFSET, gamma=GAMMA,
                             tile=TILE, y=y, x=x, shape=[TILE, TILE, 3],
                             std=round(std, 5),
                             sat_frac=round(sat_frac, 5)),
                    )
                rec = dict(path=r["path"], tiles=len(picked), white=white,
                           shape=[int(q.shape[0]), int(q.shape[1])])
                del q, satmask
            except Exception as e:
                n_err += 1
                rec = dict(path=r["path"], error=f"{type(e).__name__}: {e}")
            fm.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if i % 25 == 0:
                fm.flush()
                el = time.perf_counter() - t0
                p(f"  {i:>5,}/{len(todo):,}  {el/60:6.1f}分  {el/i:.2f}s/枚  "
                  f"タイル {writer.total:,}  残り約 {el/i*(len(todo)-i)/60:6.1f}分  "
                  f"エラー {n_err}")

    writer.close()
    el = time.perf_counter() - t0
    p(f"\n完了: {writer.total:,} タイル / shard {writer.idx} 本  "
      f"{el/60:.1f}分  エラー {n_err} 件")
    if os.path.isdir(a.shards):
        size = sum(os.path.getsize(os.path.join(a.shards, f))
                   for f in os.listdir(a.shards) if f.endswith(".tar"))
        p(f"  合計 {size/1e9:.2f} GB  "
          f"({size/max(writer.total,1)/1e6:.2f} MB/タイル)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
