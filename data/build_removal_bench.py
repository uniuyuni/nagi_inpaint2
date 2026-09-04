"""**実際の除去課題**のベンチを作る (docs/DESIGN_V2_JA.md §6 の裁定データ ②)。

**なぜ既存ベンチでは駄目か**

`cache/bench_person_val.json` は `data/masks_dict.py` の形だけの辞書を
`data/masks.py` が**無作為な位置に貼った**もので、マスクが空や林床の上に落ちる。
つまり「隠された背景を復元する」課題であって、
**「そこに居る不要な人物を消す」課題を一度も測っていなかった** (§3.1)。

このスクリプトは `data/measure_targets.py` が実写真から検出した
**実在の人物の位置・形**にマスクを置き、フル解像度で 512² を切り出す。

**正解画像は無い。** 人物は実際にそこに居るので、消した後の「正解」は存在しない。
したがって PSNR / LPIPS のような全参照指標は使えない。
判定は**盲検A/B (ユーザ本人)** と、ハルシネーション検出で行う (§6.1)。
これは PROVE が「除去は one-to-many なので全参照指標は不適」と結論したのと同じ立場。

**マスクの作り方**

検出マスクはプレビュー解像度 (長辺 1024) なので、フル解像度へ拡大すると粗い。
運用でもユーザは対象より少し大きくマスクを描くので、
**拡大 + 膨張**で「実運用に近い、少し大きめのマスク」にする (§4.1 の方針と同じ)。
LaMa と YOEO には同一のマスクを渡すので比較は公平である。

**読み出し専用**: `/Volumes/PHOTOS 01` には一切書き込まない。
torch は import しない (libraw_enhanced と OpenMP が衝突するため)。

  pixi run python data/build_removal_bench.py --n 48
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = "/Volumes/PHOTOS 01/Photos"


def p(*a):
    print(*a, flush=True)


@contextlib.contextmanager
def quiet_fd():
    old = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(old, 1)
        os.close(devnull)
        os.close(old)


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    """半径 r の膨張 (numpy のみ)。"""
    out = m.copy()
    for _ in range(r):
        q = out
        out = (q | np.roll(q, 1, 0) | np.roll(q, -1, 0)
               | np.roll(q, 1, 1) | np.roll(q, -1, 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="cache/targets.jsonl")
    ap.add_argument("--masks", default="cache/targets_masks.npz")
    ap.add_argument("--out", default="cache/bench_removal.npz")
    ap.add_argument("--crop", type=int, default=512,
                    help="原寸で切り出す一辺。512 より大きくすると"
                         "**視野が広がる代わりに、モデルへ渡すとき縮小される**")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-rel", type=float, default=0.05,
                    help="画面高さに対する対象の高さの上限。既定は主用途の「小さい人」")
    ap.add_argument("--dilate", type=int, default=6, help="マスクの膨張半径 (フル解像度 px)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import libraw_enhanced as lre
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract import GAMMA, POSTPROCESS_KW, WHITE_PCT

    rows = [json.loads(l) for l in open(a.targets)]
    z = np.load(a.masks, allow_pickle=False)
    mk, shp = z["masks"], z["shapes"]

    # 主用途「小さい人」に絞る。§3.1.5 の実測で 512² に 100% 収まる帯。
    sel = [r for r in rows if r["bh"] / r["full_h"] <= a.max_rel
           and max(r["bh"], r["bw"]) <= a.crop * 0.75]
    p(f"対象 {len(rows)} 件中 {len(sel)} 件が主用途の帯 "
      f"(画面高さ {a.max_rel*100:.0f}% 以下、{a.crop}² に収まる)")

    rng = np.random.default_rng(a.seed)
    # 同じ写真から偏らないよう、1 枚につき最大 2 件
    per: dict[str, int] = {}
    picked = []
    for i in rng.permutation(len(sel)):
        r = sel[int(i)]
        if per.get(r["path"], 0) >= 2:
            continue
        per[r["path"]] = per.get(r["path"], 0) + 1
        picked.append(r)
        if len(picked) >= a.n:
            break
    p(f"{len(picked)} 件を採用 (元写真 {len(per)} 枚)")

    C = a.crop
    imgs = np.zeros((len(picked), C, C, 3), np.float16)
    msks = np.zeros((len(picked), C, C), np.uint8)
    meta = []
    n = 0
    t0 = time.perf_counter()
    cache_path, cache_img, cache_white = None, None, 1.0

    for r in picked:
        path = os.path.join(ROOT, r["path"])
        if not path.lower().endswith(".raf"):
            path += ".RAF" if os.path.exists(path + ".RAF") else ""
        if not os.path.exists(path):
            p(f"  [skip] 見つからない: {r['path']}")
            continue
        try:
            if cache_path != path:          # 同じ写真から 2 件取るときの再現像を避ける
                with quiet_fd():
                    with lre.imread(path) as raw:
                        cache_img = np.asarray(raw.postprocess(**POSTPROCESS_KW))
                # **white は画像全体で一度だけ決める** (data/extract.py と同じ規約)。
                # クロップごとに決めると shard の作り方とずれる。
                w = float(np.percentile(cache_img[..., :3].mean(axis=2), WHITE_PCT))
                cache_white = w if (np.isfinite(w) and w > 0) else 1.0
                cache_path = path
            full, white = cache_img, cache_white
        except Exception as e:
            p(f"  [skip] {os.path.basename(path)}: {type(e).__name__}")
            continue

        H, W = full.shape[:2]
        # 検出マスク (プレビュー解像度) をフル解像度へ最近傍で拡大
        m_small = np.unpackbits(mk[r["mi"]])[:shp[r["mi"], 0] * shp[r["mi"], 1]]
        m_small = m_small.reshape(shp[r["mi"], 0], shp[r["mi"], 1]).astype(bool)
        yi = np.clip((np.arange(H) * m_small.shape[0] / H).astype(np.int32),
                     0, m_small.shape[0] - 1)
        xi = np.clip((np.arange(W) * m_small.shape[1] / W).astype(np.int32),
                     0, m_small.shape[1] - 1)

        cy, cx = int(round(r["cy"] * H / r["full_h"])), int(round(r["cx"] * W / r["full_w"]))
        y0 = int(np.clip(cy - C // 2, 0, max(0, H - C)))
        x0 = int(np.clip(cx - C // 2, 0, max(0, W - C)))
        crop = full[y0:y0 + C, x0:x0 + C]
        if crop.shape[0] != C or crop.shape[1] != C:
            continue
        cm = m_small[yi[y0:y0 + C]][:, xi[x0:x0 + C]]  # noqa: E501
        cm = dilate(cm, a.dilate)
        if cm.sum() < 100 or cm.mean() > 0.5:
            continue

        c3 = crop[..., :3]
        pc = np.sign(c3) * np.power(np.abs(c3) / white, 1.0 / GAMMA)
        imgs[n] = pc.astype(np.float16)
        msks[n] = cm.astype(np.uint8)
        meta.append(dict(path=r["path"], score=r["score"], bh=r["bh"], bw=r["bw"],
                         rel=r["bh"] / r["full_h"], white=float(white),
                         y0=y0, x0=x0, area=float(cm.mean())))
        n += 1
        if n % 8 == 0:
            el = time.perf_counter() - t0
            p(f"  {n}/{len(picked)}  {el/n:.1f} 秒/件")

    imgs, msks = imgs[:n], msks[:n]
    np.savez_compressed(a.out, img=imgs, mask=msks,
                        meta=json.dumps(meta, ensure_ascii=False), crop=C)
    el = time.perf_counter() - t0
    p(f"\n{n} 件 -> {a.out} ({os.path.getsize(a.out)/1e6:.0f} MB, {el/60:.1f} 分)")
    if meta:
        ar = np.array([m["area"] for m in meta])
        bh = np.array([m["bh"] for m in meta])
        p(f"  マスク面積比 中央値 {np.median(ar)*100:.2f}% / "
          f"対象の高さ 中央値 {np.median(bh):.0f}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
