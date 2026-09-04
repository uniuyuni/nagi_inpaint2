"""実際の除去課題で LaMa と YOEO を比べる (docs/DESIGN_V2_JA.md §6)。

**正解画像は無い。** 人物は実際にそこに居るので、消した後の「正解」は存在しない。
PROVE が結論したとおり、除去は one-to-many であり全参照指標 (PSNR/LPIPS) は不適で、
しかも「周囲からのコピー貼り付け」を高く評価してしまう。

そこで**正解を必要としない 3 つの軸**で測る:

1. **残存検出** — 出力にもう一度 Mask R-CNN を掛け、マスク領域に人物が残っていないか。
   ユーザの判断基準「消したいものが消えているか」を直接測る唯一の自動指標。
2. **ハルシネーション** — マスク領域に**新しい**物体 (人物以外も含む) が出ていないか。
   YOEO は "sundries suppression loss" でこれを抑えて学習しているので、差が出るはず。
3. **周囲との整合** — 穴の内側と周囲の特徴分布の距離 (RC-S 相当)。
   **選別には使わない。報告のみ** (§6.1 の Goodhart 対策)。

**合否は盲検A/B (ユーザ本人) で決める。** 上の 3 つは補助である。

  pixi run python eval/removal_compare.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import grain_ratio                              # noqa: E402
from eval.perceptual import Perceptual                            # noqa: E402


def p(*a):
    print(*a, flush=True)


def u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def detect_in(det, img: np.ndarray, mask: np.ndarray, device: str,
              score: float = 0.5):
    """マスク領域に何が写っているかを調べる。

    戻り値: (人物の残存スコア, その他物体の最大スコア)

    **重心がマスク内にあるものだけを数える。** 初版は「検出の 50% 以上が
    マスク内」としていたが、消したい対象の**隣に居る別の人物**を拾ってしまい、
    除去に成功しているのに「残っている」と報告した (実例: 撮影者の隣のバッグを
    消した場面で、残っている撮影者本人を拾って 1.00 を付けた)。
    重心条件にすると「マスクの位置に何かが居るか」だけを見ることになる。
    """
    t = torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1)).float().to(device)
    with torch.inference_mode():
        o = det([t])[0]
    person = other = 0.0
    for i in range(len(o["scores"])):
        s = float(o["scores"][i])
        if s < score:
            continue
        m = (o["masks"][i, 0] > 0.5).cpu().numpy()
        if m.sum() < 64:
            continue
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        if not mask[cy, cx]:
            continue                       # 重心がマスクの外 = 別の物体
        if (m & mask).sum() / m.sum() < 0.3:
            continue                       # ほとんどマスク外に伸びている = 別の物体
        if int(o["labels"][i]) == 1:
            person = max(person, s)
        else:
            other = max(other, s)
    return person, other


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="cache/bench_removal.npz")
    ap.add_argument("--out", default="out/removal")
    ap.add_argument("--n", type=int, default=0, help="0 なら全件")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                              maskrcnn_resnet50_fpn_v2)
    from eval.lama_baseline import Lama
    from nagi_inpaint2.removers import Yoeo

    z = np.load(a.bench, allow_pickle=False)
    imgs, msks = z["img"], z["mask"]
    meta = json.loads(str(z["meta"]))
    n = len(meta) if a.n <= 0 else min(a.n, len(meta))
    p(f"実除去ベンチ {n} 件 (正解画像なし)")

    det = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=512, max_size=512).eval().to(a.device)
    lama = Lama(device=a.device)
    yoeo = Yoeo(device=a.device, steps=2)
    perc = Perceptual(device=a.device)
    os.makedirs(a.out, exist_ok=True)

    rows = []
    tl = ty = 0.0
    for i in range(n):
        img = imgs[i].astype(np.float32)
        mask = msks[i].astype(bool)

        t = time.perf_counter(); fl = lama(img, mask); tl += time.perf_counter() - t
        t = time.perf_counter(); fy = yoeo(img, mask); ty += time.perf_counter() - t
        ol = np.where(mask[..., None], fl, img)
        oy = np.where(mask[..., None], fy, img)

        # 1. 元画像に人物が居ることを確認 (陰性対照。居なければベンチが壊れている)
        p0, _ = detect_in(det, img, mask, a.device)
        pl, hl = detect_in(det, ol, mask, a.device)
        py, hy = detect_in(det, oy, mask, a.device)

        rows.append(dict(
            i=i, path=meta[i]["path"], bh=meta[i]["bh"], area=meta[i]["area"],
            src_person=p0, lama_person=pl, yoeo_person=py,
            lama_other=hl, yoeo_other=hy,
            lama_coh=perc(ol, img, ~mask), yoeo_coh=perc(oy, img, ~mask),
            lama_grain=grain_ratio(ol, mask), yoeo_grain=grain_ratio(oy, mask)))

        H, W = img.shape[:2]
        sheet = np.full((H, W * 3 + 8, 3), 32, np.uint8)
        for j, pan in enumerate((img, ol, oy)):
            sheet[:, j * (W + 4):j * (W + 4) + W] = u8(pan)
        Image.fromarray(sheet).save(
            os.path.join(a.out, f"{i:02d}_{os.path.basename(meta[i]['path'])}.png"))
        p(f"  {i+1}/{n}  元{p0:.2f} -> LaMa {pl:.2f} / YOEO {py:.2f}")

    f = lambda k: float(np.median([r[k] for r in rows if np.isfinite(r[k])]))  # noqa: E731
    p(f"\n=== 実除去ベンチ {len(rows)} 件 (列: 元画像 | LaMa | YOEO) ===\n")
    p(f"{'':10s} {'人物の残存↓':>12s} {'新規物体↓':>10s} {'周囲整合↓':>10s} "
      f"{'grain':>8s} {'秒/枚':>7s}")
    p(f"{'元画像':10s} {f('src_person'):12.3f} {'':>10s} {'':>10s}")
    p(f"{'LaMa':10s} {f('lama_person'):12.3f} {f('lama_other'):10.3f} "
      f"{f('lama_coh'):10.5f} {f('lama_grain'):8.3f} {tl/max(n,1):7.2f}")
    p(f"{'YOEO':10s} {f('yoeo_person'):12.3f} {f('yoeo_other'):10.3f} "
      f"{f('yoeo_coh'):10.5f} {f('yoeo_grain'):8.3f} {ty/max(n,1):7.2f}")

    for nm in ("lama", "yoeo"):
        left = sum(1 for r in rows if r[f"{nm}_person"] > 0.5)
        hal = sum(1 for r in rows if r[f"{nm}_other"] > 0.5)
        p(f"\n  {nm.upper()}: 人物が残った {left}/{len(rows)} 件 / "
          f"新規物体が出た {hal}/{len(rows)} 件")
    src_ok = sum(1 for r in rows if r["src_person"] > 0.5)
    p(f"\n  陰性対照: 元画像で人物を検出できた {src_ok}/{len(rows)} 件 "
      f"(低ければベンチかマスクが壊れている)")
    json.dump(rows, open(os.path.join(a.out, "rows.json"), "w"),
              ensure_ascii=False, indent=1)
    p(f"\n{a.out} に比較画像。**必ず目で見ること** (R16)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
