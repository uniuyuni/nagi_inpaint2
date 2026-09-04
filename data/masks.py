"""マスク生成 (SPEC §6.5)。

学習時のマスク分布を実利用に寄せる。手作業のマスク実例は不要。

本モジュールは **手続き生成**を担当する:

  wire   : 電線。細長く緩やかに曲がる線。実利用で最も多い除去対象
  dust   : センサーダスト。小さな円が散在する
  stroke : LaMa 風のブラシストローク (wide / thin)
  rect   : 矩形。看板やウォーターマークの除去を模す
  blob   : 不定形の塊。人物や物体のシルエット代替

物体シルエット辞書 (data/masks_dict.py、セグメンテーションで抽出) と混合して使う。
シルエットを別の画像に貼ることで形状の多様性を自己教師的に確保する。

**マスク面積比は 0.1% 〜 8% の対数一様分布** (§6.5)。
スコープが「小〜中の除去」(§1.2) なので、面積の大きいマスクは作らない。

  pixi run python data/masks.py --preview out/masks_preview.png
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

KINDS = ("wire", "dust", "stroke", "rect", "blob")
# 実利用の頻度に寄せた重み。電線とブラシストロークが主
KIND_WEIGHTS = (0.28, 0.12, 0.32, 0.13, 0.15)

AREA_LO, AREA_HI = 0.001, 0.08

# 実物体シルエット辞書 (data/masks_dict.py が作る)。
# `masks`: packbits((N, size, size), axis=-1) の uint8、`size`: 格納解像度 (128)。
MASK_DICT_PATH = "cache/mask_dict.npz"
_mask_dict_cache: np.ndarray | None = None


def _canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("L", (size, size), 0)
    return im, ImageDraw.Draw(im)


def _bezier(pts: np.ndarray, n: int = 64) -> list[tuple[float, float]]:
    """制御点から2次/3次ベジェを離散化する。"""
    t = np.linspace(0, 1, n)[:, None]
    k = len(pts)
    out = np.zeros((n, 2))
    for i, p in enumerate(pts):
        c = math.comb(k - 1, i)
        out += c * (t ** i) * ((1 - t) ** (k - 1 - i)) * p
    return [tuple(v) for v in out]


def wire(size: int, rng: np.random.Generator) -> Image.Image:
    """電線。画面を横切る細長い線を1〜3本。太さは 1〜5px。"""
    im, d = _canvas(size)
    for _ in range(rng.integers(1, 4)):
        # 対辺から対辺へ抜ける制御点
        if rng.random() < 0.5:
            p0 = (0, rng.uniform(0, size))
            p2 = (size, rng.uniform(0, size))
        else:
            p0 = (rng.uniform(0, size), 0)
            p2 = (rng.uniform(0, size), size)
        p1 = (rng.uniform(0, size), rng.uniform(0, size))
        # 電線は緩い懸垂曲線なので、中間制御点を両端の中点寄りに引き戻す
        mid = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)
        w = rng.uniform(0.15, 0.5)
        p1 = (mid[0] * (1 - w) + p1[0] * w, mid[1] * (1 - w) + p1[1] * w)
        # 太さは解像度に比例させる (512² で 1〜5px 相当)
        w_px = max(1, int(round(size * rng.uniform(0.002, 0.010))))
        d.line(_bezier(np.array([p0, p1, p2], dtype=float)),
               fill=255, width=w_px, joint="curve")
    return im


def dust(size: int, rng: np.random.Generator) -> Image.Image:
    """センサーダスト。小さな円が散在する。"""
    im, d = _canvas(size)
    for _ in range(rng.integers(3, 25)):
        r = size * rng.uniform(0.004, 0.028)      # 512² で 2〜14px 相当
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return im


def stroke(size: int, rng: np.random.Generator) -> Image.Image:
    """LaMa 風のブラシストローク。"""
    im, d = _canvas(size)
    wide = rng.random() < 0.5
    width = int(max(1, round(size * (rng.uniform(0.04, 0.12) if wide
                                     else rng.uniform(0.004, 0.025)))))
    for _ in range(rng.integers(1, 4)):
        n = int(rng.integers(3, 6))
        pts = rng.uniform(0, size, size=(n, 2))
        d.line(_bezier(pts), fill=255, width=width, joint="curve")
    return im


def rect(size: int, rng: np.random.Generator) -> Image.Image:
    """矩形。看板やウォーターマークの除去を模す。"""
    im, d = _canvas(size)
    for _ in range(rng.integers(1, 3)):
        w = rng.uniform(size * 0.05, size * 0.35)
        h = rng.uniform(size * 0.05, size * 0.35)
        x = rng.uniform(0, size - w)
        y = rng.uniform(0, size - h)
        d.rectangle([x, y, x + w, y + h], fill=255)
    return im


def blob(size: int, rng: np.random.Generator) -> Image.Image:
    """不定形の塊。極座標で半径を緩く揺らした閉曲線。

    物体シルエット辞書 (セグメンテーション由来) が入るまでの代替なので、
    人物・看板・ゴミのように**凸で塊状**であることを優先する。
    摂動の周波数と振幅を上げるとすぐ星形・十字形になるので低く抑える。
    縦横比の異方性で「立っている人物」寄りの形も作れるようにしてある。
    """
    im, d = _canvas(size)
    cx, cy = rng.uniform(size * 0.25, size * 0.75, size=2)
    base = rng.uniform(size * 0.08, size * 0.25)
    n = 96
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    for freq, amp in ((rng.integers(2, 4), rng.uniform(0.08, 0.18)),
                      (rng.integers(3, 6), rng.uniform(0.04, 0.10))):
        r += amp * np.sin(th * freq + rng.uniform(0, 2 * np.pi))
    r = np.clip(r, 0.55, 1.6) * base
    ax, ay = 1.0, rng.uniform(0.6, 2.2)      # 縦長/横長の異方性
    pts = [(cx + rr * ax * math.cos(t), cy + rr * ay * math.sin(t))
           for t, rr in zip(th, r)]
    d.polygon(pts, fill=255)
    return im


def _load_mask_dict() -> np.ndarray:
    """`cache/mask_dict.npz` を読み込んでモジュールにキャッシュする (初回のみ)。

    フォーマット (data/masks_dict.py の保存部分を参照):
      masks : np.packbits(bool 配列 (N, size, size), axis=-1) -> uint8 (N, size, size/8)
      size  : 格納解像度 (128)
      shoots: 由来ショット名 (このモジュールでは未使用)
    """
    global _mask_dict_cache
    if _mask_dict_cache is None:
        if not os.path.exists(MASK_DICT_PATH):
            raise FileNotFoundError(
                f"{MASK_DICT_PATH} が見つかりません。silhouette 種別を使うには、"
                "先に `pixi run python data/masks_dict.py` で実物体シルエット辞書を"
                "作成してください (MobileSAM で自前タイルから抽出する、CPU 専用の"
                "重い処理です)。他の種別 (wire/dust/stroke/rect/blob) には影響しません。"
            )
        d = np.load(MASK_DICT_PATH)
        store = int(d["size"])
        m = np.unpackbits(d["masks"], axis=-1)[:, :, :store]
        _mask_dict_cache = m > 0
    return _mask_dict_cache


def silhouette(size: int, rng: np.random.Generator) -> Image.Image:
    """実物体シルエット。`cache/mask_dict.npz` (data/masks_dict.py が MobileSAM で
    自前タイルから抽出した実シルエット) から1枚ランダムに選び、等方スケール・
    左右反転・小さな回転のみを施してランダム位置に配置する。

    **なぜ手続き生成 (blob) では不十分か**: `blob` は極座標で半径を緩く揺らした
    閉曲線にすぎず、常に凸に近い一塊の形にしかならない。しかし本用途の主対象
    (小さく写った人物) の輪郭は、頭部+肩のくびれ・腕や脚のような細い突起・裾や
    髪による局所的な凹みなど、手続き的なパラメータ数個では再現できない不規則性
    を持つ。この不規則性 (充填率や凹凸の統計) を再現するには、実際のセグメン
    テーション結果を直接貼るのが最も確実。

    人物は縦長なので、アスペクト比を大きく崩す異方スケールは行わない
    (等方スケール + 左右反転のみを基本とし、回転はごく小角度に留める)。
    面積比の調整は `sample_mask` の再抽選機構に委ねる。
    """
    masks = _load_mask_dict()
    idx = int(rng.integers(len(masks)))
    im = Image.fromarray((masks[idx].astype(np.uint8)) * 255)

    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

    # 回転は任意。人物の縦長さを保つため小角度のみ
    angle = rng.uniform(-15.0, 15.0)
    if abs(angle) > 1e-3:
        im = im.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=0)

    # 等方スケール。辞書内の画像は物体の外接矩形の長辺が格納解像度いっぱいに
    # なるよう正規化済みなので、長辺がキャンバスの 12%〜40% になるよう縮尺する。
    # 対数一様にするのは、sample_mask の目標面積も対数一様 (§6.5) であり、
    # 小面積側の的中率を稼ぐため (線形一様だと大きい側に偏る)
    src_w, src_h = im.size
    max_side = max(src_w, src_h)
    u = float(np.exp(rng.uniform(math.log(0.12), math.log(0.40))))
    scale = size * u / max_side
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    im = im.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (size, size), 0)
    x = int(rng.integers(0, size - new_w + 1))
    y = int(rng.integers(0, size - new_h + 1))
    canvas.paste(im, (x, y))
    return canvas


GENERATORS = dict(wire=wire, dust=dust, stroke=stroke, rect=rect, blob=blob,
                  silhouette=silhouette)


def sample_mask(size: int = 512, rng: np.random.Generator | None = None,
                kind: str | None = None, target_area: float | None = None,
                max_tries: int = 6) -> tuple[np.ndarray, dict]:
    """マスクを1枚生成する。面積比が目標に収まるまで数回引き直す。

    戻り値は (bool 配列 (size, size), メタ情報)。
    """
    rng = rng or np.random.default_rng()
    if kind is None:
        kind = str(rng.choice(KINDS, p=KIND_WEIGHTS))
    if target_area is None:
        # 対数一様
        target_area = float(np.exp(rng.uniform(math.log(AREA_LO), math.log(AREA_HI))))

    best = None
    for _ in range(max_tries):
        m = np.asarray(GENERATORS[kind](size, rng), dtype=np.uint8) > 127
        area = float(m.mean())
        if best is None or abs(area - target_area) < abs(best[1] - target_area):
            best = (m, area)
        if AREA_LO <= area <= AREA_HI and 0.5 * target_area <= area <= 2 * target_area:
            break
    m, area = best
    return m, dict(kind=kind, area=round(area, 5), target_area=round(target_area, 5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default="out/masks_preview.png")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import collections
    import os

    rng = np.random.default_rng(a.seed)

    # 面積分布の確認
    areas = collections.defaultdict(list)
    for _ in range(2000):
        _, meta = sample_mask(a.size, rng)
        areas[meta["kind"]].append(meta["area"])
    print(f"{'種別':<8}{'n':>6}{'面積 p10':>10}{'p50':>9}{'p90':>9}")
    allv = []
    for k in KINDS:
        v = areas[k]
        allv += v
        if v:
            q = np.percentile(v, [10, 50, 90])
            print(f"{k:<8}{len(v):>6}{q[0]:>10.4f}{q[1]:>9.4f}{q[2]:>9.4f}")
    q = np.percentile(allv, [10, 50, 90])
    print(f"{'全体':<8}{len(allv):>6}{q[0]:>10.4f}{q[1]:>9.4f}{q[2]:>9.4f}"
          f"   (目標 {AREA_LO}-{AREA_HI} の対数一様)")

    # プレビュー
    cols, rows = 8, len(KINDS)
    sheet = Image.new("L", (a.size * cols, a.size * rows), 0)
    for r, k in enumerate(KINDS):
        for c in range(cols):
            m, _ = sample_mask(a.size, rng, kind=k)
            sheet.paste(Image.fromarray((m * 255).astype(np.uint8)),
                        (c * a.size, r * a.size))
    os.makedirs(os.path.dirname(a.preview) or ".", exist_ok=True)
    sheet.save(a.preview)
    print(f"\nプレビュー: {a.preview}  (行=種別 {', '.join(KINDS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
