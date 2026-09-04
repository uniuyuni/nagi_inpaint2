"""リニアHDR / 広色域の軸で LaMa との差を測る (SPEC §1.3 の残る差別化要因)。

**なぜこれが最後の砦か**

解像度軸は崩れた (§3 冒頭の訂正)。LaMa はタイル分割で 40MP を 10秒・
boundary 1.18 で処理できてしまう。粒状も継ぎ目も既に良い。

残るのは「LaMa は 8bit sRGB 専用である」という**構造的制約**だけである。
本プロジェクトは ProPhoto リニア float32 のまま処理する。
この差が実測でどれだけあるかを、ここで確定させる。

**これまでの測定はこの差を消していた**: eval/lama_baseline.py は
知覚空間の値をタイルごとに [0,1] へ正規化して LaMa に渡していた。
インペインティング能力だけを比較するには正しいが、
**実運用では 8bit sRGB への往復が必ず入る**。それを含めて測るのが本スクリプト。

比較する経路:

  float直接   知覚空間を正規化して LaMa へ (実運用では不可能。LaMa の上限値)
  sRGB8bit    ProPhoto リニア -> sRGB -> ガンマ -> 8bit -> LaMa -> 逆変換
              (LaMa を実際のワークフローに入れたときに起きること)
  往復のみ    インペイントせず色経路だけ通す (経路そのものの損失の下限)

  pixi run python eval/hdr_test.py --n 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.lama_baseline import Lama                  # noqa: E402
from eval.metrics import psnr, report                # noqa: E402
from eval.validate_stage_d import load_tiles         # noqa: E402
from masks import sample_mask                        # noqa: E402
from nagi_inpaint2.color import to_linear, to_perceptual   # noqa: E402

# ProPhoto RGB (ROMM, D50 白色点) -> XYZ(D50)
M_PROPHOTO_XYZ = np.array([
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100]])
# XYZ(D65) -> sRGB
M_XYZ_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]])
# Bradford D50 -> D65
M_BRADFORD_D50_D65 = np.array([
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098]])

M_PP_TO_SRGB = M_XYZ_SRGB @ M_BRADFORD_D50_D65 @ M_PROPHOTO_XYZ
M_SRGB_TO_PP = np.linalg.inv(M_PP_TO_SRGB)


def srgb_encode(x: np.ndarray) -> np.ndarray:
    """リニア -> sRGB 伝達関数 (区分定義)。"""
    a = 0.0031308
    return np.where(x <= a, x * 12.92, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def srgb_decode(x: np.ndarray) -> np.ndarray:
    a = 0.04045
    return np.where(x <= a, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def to_srgb8(lin_pp: np.ndarray) -> tuple[np.ndarray, dict]:
    """リニア ProPhoto -> sRGB 8bit。何が失われたかも返す。"""
    lin_s = lin_pp @ M_PP_TO_SRGB.T
    below = float((lin_s < 0).any(axis=2).mean())      # 色域外 (彩度が高すぎる)
    above = float((lin_s > 1).any(axis=2).mean())      # 白飛び (ハイライト)
    enc = srgb_encode(np.clip(lin_s, 0, 1))
    u8 = np.clip(np.rint(enc * 255), 0, 255).astype(np.uint8)
    return u8, dict(oog=below, clip=above)


def from_srgb8(u8: np.ndarray) -> np.ndarray:
    """sRGB 8bit -> リニア ProPhoto。"""
    return (srgb_decode(u8.astype(np.float32) / 255.0) @ M_SRGB_TO_PP.T).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    shards = sorted(f for f in os.listdir(a.shards) if f.endswith(".tar"))
    tiles = load_tiles(os.path.join(a.shards, shards[0]), a.n)
    lama = Lama(device=a.device)
    rng = np.random.default_rng(a.seed)
    print(f"タイル {len(tiles)} 件\n")

    acc = {k: [] for k in ("LaMa float直接", "LaMa sRGB8bit経路", "色経路の往復のみ")}
    stats = []
    rt_psnr = []

    for p_img, satm, meta in tiles:
        mask, _ = sample_mask(meta["tile"], rng)
        if mask.sum() < 256:
            continue
        white = meta["white"]
        lin = to_linear(p_img, white)

        # --- 色経路そのものの損失 (インペイントなし) ---
        u8, st = to_srgb8(lin)
        stats.append(st)
        rt_lin = from_srgb8(u8)
        rt_p = to_perceptual(rt_lin, white)
        rt_psnr.append(psnr(rt_p, p_img))
        acc["色経路の往復のみ"].append(report(rt_p, p_img, mask, gt=p_img))

        # --- LaMa: 知覚空間 float を直接 (実運用では不可能な上限値) ---
        f_direct = lama(p_img, mask)
        acc["LaMa float直接"].append(
            report(np.where(mask[..., None], f_direct, p_img), p_img, mask, gt=p_img))

        # --- LaMa: 8bit sRGB 経路 (実運用で起きること) ---
        x8 = srgb_decode(u8.astype(np.float32) / 255.0)      # リニア sRGB
        g8 = srgb_encode(x8)                                  # LaMa はガンマ域を期待
        f8 = lama(g8, mask)                                   # 8bit 相当のガンマ域で処理
        back_lin = srgb_decode(np.clip(f8, 0, 1)) @ M_SRGB_TO_PP.T
        back_p = to_perceptual(back_lin.astype(np.float32), white)
        acc["LaMa sRGB8bit経路"].append(
            report(np.where(mask[..., None], back_p, p_img), p_img, mask, gt=p_img))

    def med(rs, k):
        v = [x[k] for x in rs if np.isfinite(x.get(k, np.nan))]
        return float(np.median(v)) if v else float("nan")

    print("=== sRGB 8bit へ落とすときに失われるもの ===")
    print(f"  色域外になる画素 (負値クリップ): {np.mean([s['oog'] for s in stats])*100:6.2f}%")
    print(f"  白飛びする画素   (>1 クリップ): {np.mean([s['clip'] for s in stats])*100:6.2f}%")
    print(f"  色経路の往復のみの PSNR       : {np.median(rt_psnr):6.2f} dB "
          f"(インペイントせず通しただけ)")

    print(f"\n{'経路':<20}{'boundary':>10}{'grain':>8}{'psd':>9}{'psnr':>8}")
    print(f"{'理想値':<20}{'1.00':>10}{'1.00':>8}{'0.00':>9}{'高い':>8}")
    for k, rs in acc.items():
        if rs:
            print(f"{k:<20}{med(rs,'boundary'):>10.3f}{med(rs,'grain'):>8.3f}"
                  f"{med(rs,'psd'):>9.4f}{med(rs,'psnr'):>8.2f}")

    d = med(acc["LaMa float直接"], "psnr") - med(acc["LaMa sRGB8bit経路"], "psnr")
    print(f"\n8bit sRGB 経路による損失: {d:+.2f} dB")
    print("  (これが本設計の残る差別化要因の大きさ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
