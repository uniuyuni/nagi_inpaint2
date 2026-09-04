"""Phase 0b の結果を目で見る (SPEC R16: 数値だけで品質を判断しない)。

2 つの arm の checkpoint を読み、同じ保留ケースについて横並びの画像を出す:

    入力(穴) | arm gt の出力 | arm lama の出力 | LaMa (教師) | 正解

**なぜ必要か**: v1 は「知覚距離で 13% 劣位＝接近している」と報告し続け、
盲検A/Bでは 0勝16敗だった。指標が良い方向に動いても、
**実際に見て「ぼけた塊」でないことを確かめるまで結論を出さない。**

  pixi run python eval/distill_visual.py --n 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_f import StageF                          # noqa: E402
from train.distill_train import Pack                              # noqa: E402


def p(*a):
    print(*a, flush=True)


def to_u8(x: np.ndarray) -> np.ndarray:
    """知覚空間 -> 8bit。全パネルで同じスケールを使う (比較のため)。"""
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load(path: str, device: str):
    if not os.path.exists(path):
        return None, None
    ck = torch.load(path, map_location=device)
    m = StageF(base=ck["base"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="cache/distill_val.npz")
    ap.add_argument("--gt-ckpt", default="checkpoints/distill_gt/last.pt")
    ap.add_argument("--lama-ckpt", default="checkpoints/distill_lama/last.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="out/distill")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    va = Pack(a.val)
    m_gt, ck_gt = load(a.gt_ckpt, a.device)
    m_la, ck_la = load(a.lama_ckpt, a.device)
    for nm, ck in (("gt", ck_gt), ("lama", ck_la)):
        if ck is None:
            p(f"警告: arm {nm} の checkpoint が無い")
        else:
            v = ck.get("val", {})
            p(f"arm {nm}: step {ck['step']}  d_target {v.get('d_target', float('nan')):.5f}"
              f"  grain {v.get('grain', float('nan')):.3f}")

    os.makedirs(a.out, exist_ok=True)
    cols = ["入力(穴)", "arm gt", "arm lama", "LaMa(教師)", "正解"]
    p(f"\n列: {' | '.join(cols)}")

    for k in range(min(a.n, va.n)):
        i, l, m = va.batch(slice(k, k + 1), a.device)
        img = i[0].permute(1, 2, 0).float().cpu().numpy()
        lam = l[0].permute(1, 2, 0).float().cpu().numpy()
        msk = m[0, 0].cpu().numpy() > 0.5
        lam_c = np.where(msk[..., None], lam, img)

        holed = img.copy()
        holed[msk] = 0.0
        panels = [holed]
        for mdl in (m_gt, m_la):
            if mdl is None:
                panels.append(np.zeros_like(img))
                continue
            with torch.inference_mode():
                o = mdl(i, m)
            c = torch.where(m > 0.5, o, i)[0].permute(1, 2, 0).float().cpu().numpy()
            panels.append(c)
        panels += [lam_c, img]

        H, W = img.shape[:2]
        sheet = np.zeros((H, W * len(panels) + 4 * (len(panels) - 1), 3), np.uint8)
        sheet[:] = 32
        for j, pan in enumerate(panels):
            x0 = j * (W + 4)
            sheet[:, x0:x0 + W] = to_u8(pan)
        meta = va.meta[k] if k < len(va.meta) else {}
        name = f"{k:02d}_{meta.get('kind', '?')}_{meta.get('key', '')}.png"
        Image.fromarray(sheet).save(os.path.join(a.out, name))
        p(f"  {name}")

    p(f"\n{a.out} に出力。**必ず目で見ること** (R16)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
