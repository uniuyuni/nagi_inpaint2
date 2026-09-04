"""評価指標 (SPEC §9.1)。

PSNR / SSIM / 知覚距離だけでは、本プロジェクトが狙う優位 (§1.3) が一切測れない。
既存手法が測ってこなかった以下の4つが主戦場になる。

  boundary_discontinuity : 継ぎ目。境界をまたぐ勾配の跳び
  grain_ratio            : 粒状の強さ。マスク内外の σ 比
  psd_distance           : 粒状の質。動径平均 PSD の距離
  spectral_distance      : 高周波の欠落 (ぼやけ)。要 GT

いずれも知覚空間 (§3.3) で計算する。
"""
from __future__ import annotations

import numpy as np


def _gray(x: np.ndarray) -> np.ndarray:
    return x[..., :3].mean(axis=2) if x.ndim == 3 else x


def _dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    m = mask.copy()
    for _ in range(r):
        o = m.copy()
        o[1:, :] |= m[:-1, :]
        o[:-1, :] |= m[1:, :]
        o[:, 1:] |= m[:, :-1]
        o[:, :-1] |= m[:, 1:]
        m = o
    return m


def boundary_discontinuity(out: np.ndarray, mask: np.ndarray,
                           annulus: int = 16) -> float:
    """継ぎ目の強さ。1.0 が「継ぎ目が見えない」。

    マスク境界をまたぐ隣接画素差の平均を、
    アニュラス内部の隣接画素差の平均で割る。
    境界だけ差が大きければ 1 より大きくなる = 継ぎ目が見えている。

    値ではなく**相対**で見るのは、画像ごとにコントラストが違うため。
    """
    g = _gray(out)
    m = mask.astype(bool)
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))

    # 境界をまたぐペア = 片側だけがマスク内
    ch = m[:, :-1] ^ m[:, 1:]
    cv = m[:-1, :] ^ m[1:, :]
    cross = np.concatenate([dh[ch], dv[cv]])
    if cross.size < 8:
        return float("nan")

    # 参照 = アニュラス内部の隣接差 (どちらもマスク外)
    ann = _dilate(m, annulus) & ~m
    rh = ann[:, :-1] & ann[:, 1:]
    rv = ann[:-1, :] & ann[1:, :]
    ref = np.concatenate([dh[rh], dv[rv]])
    if ref.size < 8 or ref.mean() < 1e-9:
        return float("nan")
    return float(cross.mean() / ref.mean())


def _highpass(x: np.ndarray) -> np.ndarray:
    """3x3 箱平均を引いた最細ディテール帯。ノイズが支配的になる。"""
    k = np.ones(3) / 3.0
    b = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, x)
    b = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, b)
    return x - b


def grain_ratio(out: np.ndarray, mask: np.ndarray, annulus: int = 32) -> float:
    """粒状の強さの比。1.0 が「周囲と同じ粒状」。

    0 に近い = マスク内だけツルツル (LaMa や拡散モデルの典型的な失敗)。
    """
    g = _highpass(_gray(out))
    m = mask.astype(bool)
    ann = _dilate(m, annulus) & ~m
    # 境界の勾配を拾わないよう、マスクは少し内側に寄せる
    inner = m & ~_dilate(~m, 3)
    if inner.sum() < 64 or ann.sum() < 64:
        return float("nan")
    so, si = float(g[ann].std()), float(g[inner].std())
    return float(si / so) if so > 1e-12 else float("nan")


def _radial_psd(x: np.ndarray, nbins: int = 24) -> np.ndarray:
    """動径平均パワースペクトル (平均1に正規化)。"""
    n = min(x.shape)
    n = n - (n % 2)
    if n < 16:
        return np.zeros(nbins)
    x = x[:n, :n]
    win = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2((x - x.mean()) * win))) ** 2
    cy = cx = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (n / 2)
    out = np.array([P[(r >= i / nbins) & (r < (i + 1) / nbins)].mean()
                    if ((r >= i / nbins) & (r < (i + 1) / nbins)).any() else 0.0
                    for i in range(nbins)])
    s = out.mean()
    return out / s if s > 1e-20 else out


def psd_distance(out: np.ndarray, mask: np.ndarray, annulus: int = 32) -> float:
    """粒状の「粗さ」の距離。0 が完全一致。

    σ を合わせただけでは 0 にならない。X-Trans デモザイク + defringe + CA補正が
    ノイズに与える空間相関まで一致して初めて 0 に近づく (§4.4)。
    """
    g = _highpass(_gray(out))
    m = mask.astype(bool)
    ys, xs = np.nonzero(m)
    if len(ys) < 256:
        return float("nan")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if min(y1 - y0, x1 - x0) < 16:
        return float("nan")

    ann = _dilate(m, annulus) & ~m
    ay, ax = np.nonzero(ann)
    ay0, ay1, ax0, ax1 = ay.min(), ay.max() + 1, ax.min(), ax.max() + 1

    pin = _radial_psd(g[y0:y1, x0:x1])
    pan = _radial_psd(g[ay0:ay1, ax0:ax1])
    if pin.sum() == 0 or pan.sum() == 0:
        return float("nan")
    return float(np.abs(pin - pan).mean())


def spectral_distance(out: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """高周波の欠落 (ぼやけ) を検出する。要 GT。0 が一致。"""
    m = mask.astype(bool)
    ys, xs = np.nonzero(m)
    if len(ys) < 256:
        return float("nan")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if min(y1 - y0, x1 - x0) < 16:
        return float("nan")
    a = _radial_psd(_gray(out)[y0:y1, x0:x1])
    b = _radial_psd(_gray(gt)[y0:y1, x0:x1])
    return float(np.abs(np.log1p(a) - np.log1p(b)).mean())


def psnr(out: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None,
         peak: float = 1.0) -> float:
    d = (out - gt) ** 2
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() == 0:
            return float("nan")
        d = d[m]
    mse = float(d.mean())
    return float("inf") if mse <= 0 else float(10 * np.log10(peak ** 2 / mse))


def report(out: np.ndarray, img: np.ndarray, mask: np.ndarray,
           gt: np.ndarray | None = None) -> dict:
    """まとめて計算する。gt があれば GT 必要な指標も含める。"""
    r = dict(
        boundary=boundary_discontinuity(out, mask),
        grain=grain_ratio(out, mask),
        psd=psd_distance(out, mask),
    )
    if gt is not None:
        r["psnr"] = psnr(out, gt, mask)
        r["spectral"] = spectral_distance(out, gt, mask)
    return r
