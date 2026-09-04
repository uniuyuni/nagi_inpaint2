"""ShadeWave から呼ぶための入口。**`helpers/ri_helper.py` の契約に合わせる。**

    pipe, device = setup(device="mps")
    image = predict_helper(pipe, image, mask, bbox)   # in-place 更新して同じ配列を返す

**契約** (ShadeWave `effects.py` の `InpaintEffect.make_diff` から読み取った)

  - `image` : (H, W, 3) float32 RGB、**リニア**。**1.0 を超えてよい** (HDR)。
              書き込み可能であること (in-place で更新するため)。
  - `mask`  : (H, W) または (H, W, 1) float32。**1 が除去対象。**
              **画像全体のマスク**が毎回渡される。
  - `bbox`  : `(x, y, w, h)`。**y,x ではなく x,y の順。**
              `mask_editor_touch_up` が `core.get_multiple_mask_bbox` で
              **塗った箇所ごとに分割済み**で、呼び出し側が箇所の数だけ回す。

**bbox は必ず尊重すること。** 渡されるマスクは全箇所を含むので、
bbox を無視して全体の外接矩形を取ると、離れた 2 箇所を塗ったときに
**画面全体が 1 つのクロップになり、縮小 10 倍で L4 経路に落ちる**
(実測でその経路は原寸に 12 対 1 で負ける)。**この呼び出しは bbox の中だけを消す。**

**色空間**: 入力はリニアなので、`nagi_inpaint2.perceptual_space` で知覚空間へ写し、
処理後に厳密に逆写像して返す。`white` は画像全体から求める (SPEC §3.3 の規約)。

**候補は 2 つ返せる** (`predict_candidates`)。実測 (`eval/accept.py`、全 60 件):

    LaMa 単独       63.3%  [50.7%, 74.4%]   0.63 秒
    YOEO+L5 単独    75.0%  [62.8%, 84.2%]   5.3 秒
    2 案            88.3%  [77.8%, 94.2%]   5.9 秒

**+13.3 点で +0.63 秒。** ただし ShadeWave の現行 UI は結果を 1 枚しか受け取らない
(`_set_diff_list_from_result(result_image)`) ので、
`predict_helper` は契約どおり 1 枚返す (既定は採用率の高い YOEO)。
2 案を出すには UI 側の対応が要る。
"""
from __future__ import annotations

import numpy as np

from nagi_inpaint2.perceptual_space import from_perceptual, to_perceptual, white_point

DEFAULT_CORES = ("lama", "yoeo")
DEFAULT_SINGLE = "yoeo"          # 単一で返すときは採用率の高い方
MARGIN = 0.25                    # nagi_inpaint2.pipeline.crop_box と同じ
WHITE_STRIDE = 8                 # 40MP で 0.42 秒 -> 0.005 秒、値の差 0.053%


def setup(device: str = "mps", cores=DEFAULT_CORES, crop: int = 512, **kw):
    """`(pipe, device)` を返す (`ri_helper.setup` と同じ形)。"""
    import torch
    from nagi_inpaint2.pipeline import MultiInpainter

    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    return MultiInpainter(cores=tuple(cores), device=device, crop=crop, **kw), device


# ----------------------------------------------------------------------
def _as_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return m


def _region(shape, bbox, crop: int) -> tuple[int, int, int]:
    """bbox を含む正方形を決める。`pipeline.crop_box` と同じ規則。"""
    H, W = shape[:2]
    x, y, w, h = (int(v) for v in bbox)
    need = int(round(max(h, w) * (1.0 + 2 * MARGIN)))
    side = min(max(crop, need), H, W)
    cy, cx = y + h // 2, x + w // 2
    y0 = int(np.clip(cy - side // 2, 0, H - side))
    x0 = int(np.clip(cx - side // 2, 0, W - side))
    return y0, x0, side


def _check(image: np.ndarray) -> None:
    if not (isinstance(image, np.ndarray) and image.dtype == np.float32
            and image.flags.writeable):
        raise ValueError("image は書き込み可能な float32 ndarray であること")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"image は (H, W, 3) であること: {image.shape}")


def _apply(pipe, image, mask, bbox, cores, white=None):
    """内部: パッチを作って貼る位置まで解決する。"""
    _check(image)
    m_all = _as_mask(mask)
    if m_all.shape != image.shape[:2]:
        raise ValueError(
            f"mask の形が画像と違う: {m_all.shape} 対 {image.shape[:2]}")
    y0, x0, side = _region(image.shape, bbox, pipe.crop)
    sub_lin = np.ascontiguousarray(image[y0:y0 + side, x0:x0 + side, :3])

    bx, by, bw, bh = (int(v) for v in bbox)
    sm = np.zeros((side, side), bool)
    iy0, ix0 = max(by, y0), max(bx, x0)
    iy1, ix1 = min(by + bh, y0 + side), min(bx + bw, x0 + side)
    if iy1 > iy0 and ix1 > ix0:
        sm[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = m_all[iy0:iy1, ix0:ix1] > 0.5
    if not sm.any():
        return None

    if white is None:
        white = white_point(image, WHITE_STRIDE)
    sub_p, _ = to_perceptual(sub_lin, white)
    py, px, pats = pipe.patches(sub_p, sm)

    out = {}
    for c, q in pats.items():
        if c not in cores:
            continue
        lin = from_perceptual(q, white)
        # **マスク外はリニアの原画素をそのまま戻す。**
        # 知覚空間の往復は float32 で 2.4e-07 の誤差が出るので、
        # 変換結果をそのまま貼ると SPEC §9.1 の「マスク外はビット一致」が崩れる。
        # パイプライン内部ではビット一致を守れていても、**この層で壊れる**。
        h, w = lin.shape[:2]
        hole = sm[py:py + h, px:px + w]
        src = sub_lin[py:py + h, px:px + w]
        lin[~hole] = src[~hole]
        out[c] = lin
    return y0 + py, x0 + px, out


def predict_helper(pipe, image: np.ndarray, mask: np.ndarray, bbox,
                   core: str | None = None, white: float | None = None):
    """ShadeWave 契約: **in-place 更新して同じ配列を返す。**"""
    core = core or (DEFAULT_SINGLE if DEFAULT_SINGLE in pipe.cores
                    else pipe.cores[0])
    r = _apply(pipe, image, mask, bbox, (core,), white)
    if r is None:
        return image
    ay, ax, pats = r
    q = pats[core]
    image[ay:ay + q.shape[0], ax:ax + q.shape[1], :3] = q
    return image


def predict_candidates(pipe, image: np.ndarray, mask: np.ndarray, bbox,
                       white: float | None = None) -> dict:
    """`{手法名: (y0, x0, リニアのパッチ)}`。**画像は書き換えない。**

    UI で 2 案を出すための口。貼るのは呼び出し側:

        for name, (y, x, q) in cands.items():
            out = image.copy()
            out[y:y+q.shape[0], x:x+q.shape[1], :3] = q
    """
    r = _apply(pipe, image, mask, bbox, tuple(pipe.cores), white)
    if r is None:
        return {}
    ay, ax, pats = r
    return {c: (ay, ax, q) for c, q in pats.items()}


def predict(pipe, image: np.ndarray, mask: np.ndarray,
            core: str | None = None) -> np.ndarray:
    """画像全体を 1 度に扱う口 (小さい画像用)。`ri_helper.predict` と同じ形。"""
    image = np.ascontiguousarray(image, dtype=np.float32)
    m = _as_mask(mask) > 0.5
    if not m.any():
        return image
    ys, xs = np.nonzero(m)
    bbox = (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return predict_helper(pipe, image, m.astype(np.float32), bbox, core=core)
