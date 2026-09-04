"""**消したい対象の実 px サイズを、実写真で測る** (docs/DESIGN_V2_JA.md §3.3 の M1)。

**なぜこれが要るか**

v2 初稿は「対象は中央値 78×62px なので 512² クロップで足りる」を設計の土台にしたが、
それは `data/masks.py:200` の定数 `logU(0.12, 0.40)` の言い換えに過ぎなかった (§3.1)。
撤回した結果、**対象の実サイズは未知**になり、
「512² クロップ 1 枚で完結するか」「L4 (フル解像度復元) が主経路か」が決められない。

**なぜタイルでは測れないか (実測で確認済み)**

shard のタイルは 26〜40MP からの**原寸 512² 切り出し**でシーンの文脈が無い。
COCO 学習の検出器 (800px の場面全体を前提) は、拡大された前ボケを
score 0.98 で「人物」と誤認した。ショット名で絞ると検出率 31.9% になったが
**中身は全て誤検出**だった。鮮鋭度比で落とせるが、文脈の欠落は埋まらない。

**このスクリプトの方法**

元 RAW を `half_size` で現像し (40MP -> 約 10MP)、長辺 1024 へ縮小して検出に掛ける。
**これで初めて検出器がシーン全体を見られる。** 検出した外接矩形を
**フル解像度の px に換算**して記録する。

**2 段構成にしている理由**: torch と `libraw_enhanced` は**どちらも libomp を
リンクしており、同一プロセスに同居させると OpenMP が二重初期化されて落ちる**
(`OMP: Error #15`)。`KMP_DUPLICATE_LIB_OK=TRUE` は公式に "unsafe/undocumented" と
書かれている回避策なので使わない。
現像 (libraw のみ) と検出 (torch のみ) を別プロセスに分ける。
プレビューはローカルに残るので、後で除去ベンチを組むときに再利用できる。

**読み出し専用**: `/Volumes/PHOTOS 01` には一切書き込まない。
中間生成物もサイドカーも置かない (SPEC §2 の厳守事項)。

**注意して読むこと**: 検出される人物には「意図した被写体」と
「消したい通行人」の両方が含まれる。分布はそのまま報告し、
用途に効くのは**小さい側**であることを併記する。

  pixi run python data/measure_targets.py --stage develop --n 150
  pixi run python data/measure_targets.py --stage detect
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time

import numpy as np
# **torch を module 先頭で import しない。** develop 段は libraw_enhanced のみを読み、
# detect 段だけが torch を読む。同居させると OpenMP が二重初期化されて落ちる。

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = "/Volumes/PHOTOS 01/Photos"
BODIES = ("X-E3", "X-T4", "X-T5")

# 人が写り込みやすい場所 (ショット名への部分一致、大小無視)
PLACE_WORDS = (
    "park", "shrine", "temple", "zoo", "garden", "station", "street", "city",
    "festival", "matsuri", "market", "univercity", "university", "palace",
    "tower", "paradice", "paradise", "museum", "budapest", "prague", "vienna",
    "paris", "london", "rome", "venice", "barcelona", "taipei", "seoul",
    "bangkok", "singapore", "hanoi", "tokyo", "osaka", "kyoto", "kamakura",
    "yokohama", "asakusa", "ueno",
)


def p(*a):
    print(*a, flush=True)


@contextlib.contextmanager
def quiet_fd():
    """libraw_enhanced の C++ コアが fd 1 に大量に吐くログを抑える。"""
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


def list_rafs() -> list[str]:
    out: list[str] = []
    for body in BODIES:
        b = os.path.join(ROOT, body)
        if not os.path.isdir(b):
            continue
        for year in sorted(os.listdir(b)):
            yd = os.path.join(b, year)
            if not os.path.isdir(yd):
                continue
            for shoot in sorted(os.listdir(yd)):
                if not any(w in shoot.lower() for w in PLACE_WORDS):
                    continue
                for dp, _, fs in os.walk(os.path.join(yd, shoot)):
                    out += [os.path.join(dp, f) for f in fs
                            if f.lower().endswith(".raf")]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("develop", "detect"), required=True)
    ap.add_argument("--previews", default="cache/previews.npz")
    ap.add_argument("--n", type=int, default=150, help="現像して検出する枚数")
    ap.add_argument("--det-side", type=int, default=1024, help="検出に掛ける長辺")
    ap.add_argument("--score", type=float, default=0.7)
    ap.add_argument("--out", default="cache/targets.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    if a.stage == "develop":
        return develop(a)
    return detect(a)


def develop(a) -> int:
    """段1: RAW を half_size で現像し、検出用プレビュー (uint8) を貯める。"""
    import libraw_enhanced as lre
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract import POSTPROCESS_KW

    kw = dict(POSTPROCESS_KW)
    kw["half_size"] = True                 # 検出には十分。約 4 倍速い

    rafs = list_rafs()
    random.Random(a.seed).shuffle(rafs)
    rafs = rafs[:a.n]
    p(f"対象 RAF {len(rafs)} 枚 (人が写り込みやすいショットから無作為抽出)")

    prev, meta = [], []
    t0 = time.perf_counter()
    for i, path in enumerate(rafs, 1):
        try:
            with quiet_fd():
                with lre.imread(path) as raw:
                    img = np.asarray(raw.postprocess(**kw))
        except Exception as e:
            p(f"  [skip] {os.path.basename(path)}: {type(e).__name__}")
            continue
        hh, hw = img.shape[:2]
        k = max(1e-6, float(np.percentile(img, 99.9)))
        v = np.clip(img / k, 0.0, 1.0)
        # 長辺 det_side へ縮小 (numpy のみ。torch は使わない)
        sc = a.det_side / max(hh, hw)
        nh, nw = max(1, int(round(hh * sc))), max(1, int(round(hw * sc)))
        yi = (np.arange(nh) * (hh / nh)).astype(np.int32)
        xi = (np.arange(nw) * (hw / nw)).astype(np.int32)
        small = (v[yi][:, xi] * 255 + 0.5).astype(np.uint8)
        prev.append(small)
        meta.append(dict(path=path[len(ROOT) + 1:],
                         body=path.split("/Photos/")[1].split("/")[0],
                         full_h=hh * 2, full_w=hw * 2, det_h=nh, det_w=nw))
        if i % 20 == 0:
            el = time.perf_counter() - t0
            p(f"  {i}/{len(rafs)}  {el/i:.1f} 秒/枚 / 残り約 {(len(rafs)-i)*el/i/60:.0f} 分")

    os.makedirs(os.path.dirname(a.previews) or ".", exist_ok=True)
    # サイズが揃わないので object 配列ではなく個別に詰める
    np.savez_compressed(a.previews, meta=json.dumps(meta, ensure_ascii=False),
                        **{f"p{i}": x for i, x in enumerate(prev)})
    el = time.perf_counter() - t0
    p(f"\nプレビュー {len(prev)} 枚 -> {a.previews} "
      f"({os.path.getsize(a.previews)/1e6:.0f} MB, {el/60:.1f} 分)")
    return 0


def detect(a) -> int:
    """段2: プレビューに人物検出を掛け、フル解像度 px に換算して集計する。"""
    import torch
    from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                              maskrcnn_resnet50_fpn_v2)
    det = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().to(a.device)

    z = np.load(a.previews, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    p(f"プレビュー {len(meta)} 枚")

    fo = open(a.out, "w")
    rows: list[dict] = []
    masks: list[np.ndarray] = []
    mshape: list[list[int]] = []
    n_img = n_with = 0
    t0 = time.perf_counter()

    for i, r in enumerate(meta, 1):
        small = z[f"p{i-1}"]
        n_img += 1
        t = torch.from_numpy(small.transpose(2, 0, 1)).float().div_(255.0)
        with torch.inference_mode():
            o = det([t.to(a.device)])[0]
        sel = (o["labels"] == 1) & (o["scores"] > a.score)
        idx = torch.nonzero(sel).flatten().tolist()
        if idx:
            n_with += 1
        # 検出解像度 -> フル解像度 の換算係数
        fx, fy = r["full_w"] / r["det_w"], r["full_h"] / r["det_h"]
        for j in idx:
            m = (o["masks"][j, 0] > 0.5).cpu().numpy()
            if m.sum() < 16:
                continue
            ys, xs = np.nonzero(m)
            bh = (ys.max() - ys.min() + 1) * fy
            bw = (xs.max() - xs.min() + 1) * fx
            # **位置も残す。** 除去ベンチを作るときにクロップ中心が要る。
            cy = float((ys.min() + ys.max() + 1) / 2 * fy)
            cx = float((xs.min() + xs.max() + 1) / 2 * fx)
            rec = dict(path=r["path"], body=r["body"], idx=i - 1,
                       score=float(o["scores"][j]), bh=float(bh), bw=float(bw),
                       area=float(m.mean()), full_h=r["full_h"], full_w=r["full_w"],
                       aspect=float(bh / max(bw, 1e-6)),
                       cy=cy, cx=cx, det_h=r["det_h"], det_w=r["det_w"],
                       mi=len(masks))
            masks.append(np.packbits(m.reshape(-1)))
            mshape.append([int(m.shape[0]), int(m.shape[1])])
            rows.append(rec)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        if i % 25 == 0:
            p(f"  {i}/{len(meta)}  人物 {len(rows)} 件 / 人が居た画像 {n_with} 枚")
    fo.close()
    if masks:
        mp = os.path.splitext(a.out)[0] + "_masks.npz"
        np.savez_compressed(mp, masks=np.stack(masks),
                            shapes=np.array(mshape, np.int32))
        p(f"マスク {len(masks)} 件 -> {mp}")

    el = time.perf_counter() - t0
    p(f"\n検出 {n_img} 枚 / {el/60:.1f} 分")
    p(f"人物が写っていた画像 {n_with}/{n_img} 枚 ({n_with/max(n_img,1)*100:.0f}%)")
    p(f"人物 {len(rows)} 件")
    if not rows:
        return 0

    bh = np.array([r["bh"] for r in rows])
    bw = np.array([r["bw"] for r in rows])
    ar = np.array([r["area"] for r in rows])
    asp = np.array([r["aspect"] for r in rows])
    p("\n=== 消したい対象の実サイズ (フル解像度 px 換算) ===")
    for nm, v_ in (("高さ", bh), ("幅", bw)):
        p(f"  {nm}  p10 {np.percentile(v_,10):6.0f}  中央値 {np.median(v_):6.0f}  "
          f"p90 {np.percentile(v_,90):6.0f}  最大 {v_.max():6.0f} px")
    p(f"  画面に占める面積比  中央値 {np.median(ar)*100:.2f}%  "
      f"p90 {np.percentile(ar,90)*100:.2f}%")
    p(f"  縦横比 h/w  中央値 {np.median(asp):.2f} (立ち姿の人物は通常 2.5〜3.5)")
    long_side = np.maximum(bh, bw)
    p("\n=== クロップ 1 枚に収まるか (対象の長辺で判定) ===")
    for c in (256, 384, 512, 768, 1024, 1536, 2048):
        p(f"  {c:5d}² クロップ: {(long_side <= c*0.75).mean()*100:5.1f}% が収まる "
          f"(余白 25% を見込む)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
