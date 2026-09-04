"""実写タイルから**実在の人物**を検出し、除去ベンチと対象サイズの実測に使う。

**なぜ必要か** (docs/DESIGN_V2_JA.md §3.3 の M2/M3)

v1 のマスク辞書 `data/masks_dict.py` は `SamAutomaticMaskGenerator` の全セグメントを
面積・充填率・境界接触だけで拾っており、**人物クラスの判定が無い**。
実測すると辞書 638 枚の外接矩形の縦横比は中央値 0.96 (ほぼ正方形) で、
立ち姿の人物 (通常 2.5〜3.5:1) は 20.5% しかなかった。
さらに `data/masks.py` はそれを**無作為な位置に貼る**ので、
出来上がるベンチは「空や林床の穴を復元する」課題になってしまい、
**「そこに居る不要な人物を消す」課題を一度も測っていなかった。**

このスクリプトは実タイル上で人物を検出し、**その位置・その形**のマスクを作る。
これで初めて実際の除去課題を測れる。同時に「消したい対象の実 px サイズ」
という、撤回した §3 の測定も片づく。

**検出器の選定**

SAM3 は使えなかった。キャッシュにあるのは `Sam3VideoModel` (動画追跡用) で、
かつ transformers 4.57.6 に `Sam3Model` が無い (対応は 5.x 以降だが、
diffusers を守るため 5 未満にピンしている)。
代わりに **torchvision の Mask R-CNN v2 (COCO 学習済み、person = class 1)** を使う。
人混みのタイルで 10 人を score>0.7・最高 0.996 で検出することを確認済み。

**誤検出の除去 (実測で確認)**

ショット名で絞ると検出率は 0.3% -> 31.9% に跳ね上がったが、**中身は全て誤検出だった**。
チューリップの前ボケを score 0.98 で「人物」と判定していた。
タイルは 26〜40MP からの**原寸 512² 切り出し**でシーンの文脈が無く、
COCO 学習の検出器は拡大された前ボケを大きな物体と誤認する。

**鮮鋭度比**(マスク内の高周波エネルギー / タイル全体) で分離できることを確認した:

    人混みタイル : 12 件すべて 面積 1.4〜2.0%、87〜127px、鮮鋭度比 0.79〜1.83 -> 採用
    前ボケタイル : 1 件 面積 78%、鮮鋭度比 0.19                              -> 棄却

**面積では絞らない。** 絞ると「対象は小さい」という結論をフィルタ自身が作ってしまい、
§3.1 で撤回したのと同じ誤りになる。鮮鋭度比は**誤検出という別の誤り**を落とすためのもの。

**探索の絞り込み**

無作為なタイルでの検出率は **301 枚中 1 件 (0.3%)** しかなかった。
26〜40MP から 512² を無作為に切ると人が入る確率が低いためで、検出器の問題ではない。
`cache/extract_manifest.jsonl` のショット名から**公園・寺社・動物園など
人が写り込みやすい場所**に絞ってから走査する。

  pixi run python data/find_people.py --max-tiles 12000 --out cache/people.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark import load_shard_tiles                       # noqa: E402


def hf_energy(g: np.ndarray) -> np.ndarray:
    """3x3 箱平均を引いた最細ディテール帯 (eval/metrics.py の `_highpass` と同じ流儀)。"""
    k = np.ones(3) / 3.0
    b = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, g)
    b = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, b)
    return g - b

# 人が写り込みやすい場所を示す語 (ショット名に対する部分一致、大小無視)
PLACE_WORDS = (
    "park", "shrine", "temple", "zoo", "zoolasia", "garden", "station",
    "street", "city", "festival", "matsuri", "market", "univercity",
    "university", "palace", "tower", "paradice", "paradise", "museum",
    "budapest", "prague", "vienna", "paris", "london", "rome", "venice",
    "barcelona", "taipei", "seoul", "bangkok", "singapore", "hanoi",
    "tokyo", "osaka", "kyoto", "kamakura", "yokohama", "asakusa", "ueno",
)


def p(*a):
    print(*a, flush=True)


def bench_shards() -> set[str]:
    """ベンチ用の保留 shard (学習にもベンチ生成にも混ぜない)。"""
    out: set[str] = set()
    for f in glob.glob("cache/bench_*.json") + glob.glob("cache/benchmark*.json"):
        try:
            spec = json.load(open(f))
        except Exception:
            continue
        out.update(spec.get("shards") or [])
        out.update(c["shard"] for c in spec.get("cases", []))
    return out


def shoot_map(manifest: str = "cache/extract_manifest.jsonl") -> dict[str, str]:
    """`DSCF1234` -> ショット名。タイルキーの先頭2要素から引ける形にする。"""
    m: dict[str, str] = {}
    for line in open(manifest):
        if not line.strip():
            continue
        r = json.loads(line)
        g = re.search(r"/Photos/([^/]+)/\d{4}/([^/]+)/", r["path"])
        if not g:
            continue
        body, shoot = g.group(1), g.group(2)
        base = os.path.splitext(os.path.basename(r["path"]))[0]
        m[f"{body}_{base}"] = shoot
    return m


def is_target(shoot: str | None) -> bool:
    return bool(shoot) and any(w in shoot.lower() for w in PLACE_WORDS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--out", default="cache/people.jsonl")
    ap.add_argument("--masks-out", default="cache/people_masks.npz")
    ap.add_argument("--max-tiles", type=int, default=12000, help="検出を掛けるタイル数")
    ap.add_argument("--max-hits", type=int, default=400, help="これだけ集めたら止める")
    ap.add_argument("--score", type=float, default=0.7)
    ap.add_argument("--min-area", type=int, default=200, help="マスク画素数の下限")
    ap.add_argument("--min-sharp", type=float, default=0.7,
                    help="鮮鋭度比の下限。前ボケの誤検出を落とす。**面積では絞らない**")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-shoots", action="store_true",
                    help="ショットで絞らず全タイルを対象にする")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                              maskrcnn_resnet50_fpn_v2)
    # min_size=512 にして 512² タイルの拡大を避ける (既定 800 比で 1.5倍速)
    det = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=512, max_size=512).eval().to(a.device)

    smap = shoot_map()
    held = bench_shards()
    shards = sorted(f for f in os.listdir(a.shards)
                    if f.endswith(".tar") and f not in held)
    random.Random(a.seed).shuffle(shards)
    p(f"shard {len(shards)} 本 (保留 {len(held)} 本を除外) / ショット辞書 {len(smap)} 件")

    fo = open(a.out, "w")
    hits: list[dict] = []
    masks: list[np.ndarray] = []
    n_tile = n_seen = 0
    n_blur = [0]
    t0 = time.perf_counter()
    buf: list[tuple] = []

    def flush(buf):
        nonlocal n_tile
        if not buf:
            return
        xs = [torch.from_numpy(x.transpose(2, 0, 1)).float().to(a.device)
              for _, _, x in buf]
        with torch.inference_mode():
            outs = det(xs)
        n_tile += len(buf)
        for (key, shard, v), o in zip(buf, outs):
            hp = hf_energy(v.mean(2))
            glob = float(hp.std())
            sel = (o["labels"] == 1) & (o["scores"] > a.score)
            for i in torch.nonzero(sel).flatten().tolist():
                m = (o["masks"][i, 0] > 0.5).cpu().numpy()
                if m.sum() < a.min_area:
                    continue
                sharp = float(hp[m].std()) / max(glob, 1e-9)
                if sharp < a.min_sharp:      # 前ボケの誤検出
                    n_blur[0] += 1
                    continue
                ys, xs_ = np.nonzero(m)
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs_.min()), int(xs_.max())
                h, w = m.shape
                rec = dict(shard=shard, key=key, shoot=smap.get("_".join(key.split("_")[:2])),
                           score=float(o["scores"][i]), area=float(m.mean()),
                           bh=y1 - y0 + 1, bw=x1 - x0 + 1, sharp=sharp,
                           border=bool(y0 == 0 or x0 == 0 or y1 == h - 1 or x1 == w - 1))
                hits.append(rec)
                masks.append(np.packbits(m.reshape(-1)))
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()

    for shard in shards:
        if n_tile >= a.max_tiles or len(hits) >= a.max_hits:
            break
        for key, img, _satm, _meta in load_shard_tiles(
                os.path.join(a.shards, shard), stride=1):
            n_seen += 1
            if not a.all_shoots and not is_target(smap.get("_".join(key.split("_")[:2]))):
                continue
            k = max(1.0, float(np.percentile(img, 99.9)))
            buf.append((key, shard, np.clip(img / k, 0.0, 1.0)))
            if len(buf) >= a.batch:
                flush(buf)
                buf = []
                if n_tile % 200 == 0:
                    el = time.perf_counter() - t0
                    p(f"  検出 {n_tile} 枚 (走査 {n_seen}) / 人物 {len(hits)} 件 / "
                      f"{el/max(n_tile,1):.2f} 秒/枚 / 経過 {el/60:.0f} 分")
            if n_tile >= a.max_tiles or len(hits) >= a.max_hits:
                break
    flush(buf)
    fo.close()

    if masks:
        np.savez(a.masks_out, masks=np.stack(masks), size=512,
                 meta=json.dumps(hits, ensure_ascii=False))

    el = time.perf_counter() - t0
    p(f"\n検出を掛けたタイル {n_tile} 枚 (走査 {n_seen} 枚) / {el/60:.1f} 分")
    p(f"人物 {len(hits)} 件 / タイルあたり検出率 {len(hits)/max(n_tile,1)*100:.2f}%")
    p(f"鮮鋭度比で棄却した誤検出 {n_blur[0]} 件")
    if hits:
        bh = np.array([h["bh"] for h in hits]); bw = np.array([h["bw"] for h in hits])
        ar = np.array([h["area"] for h in hits]); bd = np.array([h["border"] for h in hits])
        p(f"\n=== 消したい対象の実サイズ (§3.3 の M2) ===")
        for nm, v in (("高さ", bh), ("幅", bw)):
            p(f"  {nm}  中央値 {np.median(v):5.0f}  p90 {np.percentile(v,90):5.0f}  "
              f"最大 {v.max():4d} px")
        p(f"  面積比  中央値 {np.median(ar)*100:.2f}%  p90 {np.percentile(ar,90)*100:.2f}%")
        p(f"  縦横比 h/w  中央値 {np.median(bh/bw):.2f}  "
          f"(立ち姿の人物は通常 2.5〜3.5)")
        p(f"  タイル端に接する割合 {bd.mean()*100:.0f}%  "
          f"<- 高いほど「512²に収まらない人物」が多い")
    return 0


if __name__ == "__main__":
    sys.exit(main())
