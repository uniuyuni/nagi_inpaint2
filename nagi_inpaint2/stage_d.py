"""Stage D — 調和化 (SPEC §4.4)。学習パラメータを持たない決定的な処理。

Stage B/C が作った合成結果を、周囲と**継ぎ目なく・色が合い・粒状が一致する**状態にする。
3段階を順に適用する。順序には理由がある。

  1. 色統計整合   境界アニュラスの局所平均・共分散に合わせる
  2. 勾配ドメイン合成   境界の残差を調和膜で吸収する -> 継ぎ目が原理的に消える
  3. 粒状注入   周囲から実測した 2D PSD でノイズを合成して戻す

**なぜこの順序か**

- 色整合を勾配ドメインの**前**に置く。後に置くと、せっかく消した継ぎ目を再び作る。
- 粒状注入は勾配ドメインの**後**に置く。調和膜は滑らかな低周波成分しか足さないので、
  注入した粒状を潰さない。逆に前に置くと、膜の計算が粒状に引きずられる。

**なぜ粒状注入が要るか**

Stage C はノイズ除去済みターゲットで学習する (§6.3) ため出力がツルツルになる。
X-T5 は 40MP/APS-C で低ISOでもノイズが乗るので、そのまま貼ると
「そこだけノイズがない」不自然さが出る。これは LaMa にも拡散モデルにも無い機能であり、
実写リタッチにおける最大の差別化要因 (§1.3)。

**なぜ σ を合わせるだけでは不十分か**

X-Trans デモザイク (3-pass) + defringe + CA補正は、ノイズに強い空間相関を与える。
白色ノイズを σ だけ合わせて注入すると、周囲の「粒の粗さ」と合わない。
そこでアニュラスから 2D パワースペクトルを推定し、それでフィルタして合成する。

すべて知覚空間 (§3.3) で動作する。入力・出力とも (H, W, C) float32。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

ANNULUS = 32          # 境界アニュラスの幅 [px]
FLAT_FRAC = 0.35      # PSD 推定に使う「平坦なパッチ」の割合
PSD_PATCH = 32        # PSD 推定のパッチ辺長


# --------------------------------------------------------------------------
# 1. 境界アニュラス
# --------------------------------------------------------------------------

def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """r 回の 4近傍膨張。scipy.ndimage を使わずに済ませる。"""
    m = mask.copy()
    for _ in range(r):
        out = m.copy()
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        m = out
    return m


def annulus_of(mask: np.ndarray, width: int = ANNULUS,
               valid: np.ndarray | None = None) -> np.ndarray:
    """マスク外側のアニュラス。`valid` が与えられればそれで絞る。

    `valid` には飽和マップの否定を渡す。`highlight_mode=5` が再構成した画素は
    実測値ではなく推定値なので、色統計にもノイズ推定にも使ってはならない (§6.0)。
    """
    ann = dilate(mask, width) & ~mask
    if valid is not None:
        ann &= valid
    return ann


# --------------------------------------------------------------------------
# 2. 色統計整合
# --------------------------------------------------------------------------

def match_color(fill: np.ndarray, img: np.ndarray, mask: np.ndarray,
                ann: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """合成結果の平均・標準偏差をアニュラスに合わせる (チャンネルごと)。

    平均は後段の勾配ドメイン合成でも吸収されるが、
    **コントラスト (標準偏差) は勾配ドメインでは直らない**ため、ここで合わせる。

    統計はマスク内とアニュラスから求めるが、**補正はマスク外にも掛ける**。
    マスク内だけに掛けると fill が境界で不連続になり、その不連続を
    後段の勾配ドメイン合成が境界条件として拾って増幅してしまう
    (実際にこれを踏み、継ぎ目が 1.68 -> 4.23 に悪化した)。
    fill はマスク外にも連続に伸びていなければならない (harmonize の契約)。
    """
    if ann.sum() < 64 or mask.sum() == 0:
        return fill
    out = fill.copy()
    for c in range(fill.shape[2]):
        a = img[..., c][ann]
        f = fill[..., c][mask]
        if f.size < 8:
            continue
        sa, sf = float(a.std()), float(f.std())
        ma, mf = float(a.mean()), float(f.mean())
        gain = 1.0 if sf < 1e-8 else np.clip(sa / sf, 0.5, 2.0)
        gain = 1.0 + strength * (gain - 1.0)
        shift = strength * (ma - mf)
        out[..., c] = (fill[..., c] - mf) * gain + mf + shift
    return out


# --------------------------------------------------------------------------
# 3. 勾配ドメイン合成 (調和膜)
# --------------------------------------------------------------------------

def _membrane(mismatch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """マスク内で ∇²c = 0、境界で c = mismatch となる調和膜 c を解く。

    Poisson image editing を「補正場」で解き直した形。
    元画像と合成結果の境界での食い違いを、滑らかに内部へ拡散させて吸収する。
    値ではなく勾配を合わせるので、**継ぎ目が原理的に消える**。
    """
    h, w = mask.shape
    idx = -np.ones((h, w), dtype=np.int64)
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n == 0:
        return np.zeros_like(mismatch)
    idx[ys, xs] = np.arange(n)

    rows = [np.arange(n)]
    cols = [np.arange(n)]
    vals = [np.full(n, 4.0)]
    rhs = np.zeros(n, dtype=np.float64)

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        # 画像外は Neumann (対角を減らす) として扱う
        vals[0][~inside] -= 1.0
        ny, nx = ny[inside], nx[inside]
        src = np.nonzero(inside)[0]
        nb = idx[ny, nx]
        unknown = nb >= 0
        rows.append(src[unknown])
        cols.append(nb[unknown])
        vals.append(np.full(unknown.sum(), -1.0))
        # 既知 (マスク外) は右辺へ
        known = ~unknown
        np.add.at(rhs, src[known], mismatch[ny[known], nx[known]])

    A = csr_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    c = splu(A.tocsc()).solve(rhs)

    out = np.zeros((h, w), dtype=np.float32)
    out[ys, xs] = c
    return out


def poisson_blend(fill: np.ndarray, img: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """マスク境界の不連続を調和膜で吸収する。チャンネルごとに解く。"""
    out = fill.copy()
    if mask.sum() == 0:
        return out
    for c in range(fill.shape[2]):
        mismatch = (img[..., c] - fill[..., c]).astype(np.float64)
        out[..., c] = fill[..., c] + _membrane(mismatch, mask)
    return out


# --------------------------------------------------------------------------
# 4. 粒状注入
# --------------------------------------------------------------------------

def estimate_noise_psd(img: np.ndarray, ann: np.ndarray,
                       patch: int = PSD_PATCH) -> tuple[np.ndarray, float]:
    """アニュラスから 2D ノイズ PSD と σ を推定する。

    シーン構造が混ざらないよう、アニュラス内の**分散が小さいパッチ**だけを使う
    (平坦なパッチはノイズが支配的なので)。data/score_raw.py のノイズ床推定と同じ考え方。

    戻り値: (patch, patch) の PSD (平均1に正規化) と σ。
    """
    ys, xs = np.nonzero(ann)
    if len(ys) < patch * patch:
        return np.ones((patch, patch), dtype=np.float32), 0.0

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    gray = img[..., :3].mean(axis=2) if img.ndim == 3 else img

    # アニュラス領域を覆うパッチを集める
    win = np.hanning(patch)[:, None] * np.hanning(patch)[None, :]
    patches = []
    step = max(patch // 2, 1)
    for y in range(y0, max(y1 - patch, y0) + 1, step):
        for x in range(x0, max(x1 - patch, x0) + 1, step):
            sub = ann[y:y + patch, x:x + patch]
            if sub.shape != (patch, patch) or sub.mean() < 0.9:
                continue
            g = gray[y:y + patch, x:x + patch]
            patches.append(g)
    if not patches:
        return np.ones((patch, patch), dtype=np.float32), 0.0

    P = np.stack(patches)
    # 分散の小さい順に FLAT_FRAC を採る = ノイズ支配のパッチ
    hp = P - P.mean(axis=(1, 2), keepdims=True)
    order = np.argsort(hp.var(axis=(1, 2)))
    keep = P[order[:max(1, int(len(P) * FLAT_FRAC))]]

    acc = np.zeros((patch, patch), dtype=np.float64)
    for g in keep:
        d = (g - g.mean()) * win
        acc += np.abs(np.fft.fft2(d)) ** 2
    psd = acc / len(keep)
    psd /= max(psd.mean(), 1e-20)

    sigma = float(np.sqrt(np.median(hp[order[:max(1, int(len(P) * FLAT_FRAC))]]
                                    .var(axis=(1, 2)))))
    return psd.astype(np.float32), sigma


def synthesize_grain(shape, psd: np.ndarray, sigma: float,
                     rng: np.random.Generator) -> np.ndarray:
    """PSD に従う相関ノイズを生成する。

    白色ノイズを FFT して sqrt(PSD) を掛け戻すことで、
    X-Trans デモザイク由来の空間相関を再現する。σ だけ合わせるのでは足りない。
    """
    h, w = shape
    if sigma <= 0:
        return np.zeros((h, w), dtype=np.float32)
    ph, pw = psd.shape
    # PSD をタイル解像度へ拡大 (周波数軸の対応を保つため fftshift 経由)
    f = np.fft.fftshift(psd)
    yi = np.linspace(0, ph - 1, h)
    xi = np.linspace(0, pw - 1, w)
    f = f[np.rint(yi).astype(int)[:, None], np.rint(xi).astype(int)[None, :]]
    f = np.fft.ifftshift(f)

    white = rng.standard_normal((h, w))
    out = np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(np.maximum(f, 0))))
    s = out.std()
    return (out / s * sigma).astype(np.float32) if s > 1e-12 else out.astype(np.float32)


def _highpass(x: np.ndarray) -> np.ndarray:
    """3x3 箱平均を引いた最細ディテール帯。ノイズが支配的になる帯域。"""
    k = np.ones(3) / 3.0
    b = x
    for ax in (0, 1):
        b = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, b)
    return x - b


def estimate_grain_sigma(gray: np.ndarray, ann: np.ndarray,
                         patch: int = 16, flat_pct: float = 20.0) -> float:
    """アニュラスの粒状 σ を高周波帯で推定する。

    アニュラス全体の高周波標準偏差を使ってはならない。そこには
    **シーンのテクスチャが含まれる**ので過大評価になり、粒状を入れすぎて
    かえって継ぎ目が目立つ (実測で継ぎ目 0.88 -> 1.95 に悪化した)。

    平坦なパッチでは高周波エネルギー ≒ ノイズなので、
    アニュラス内のパッチを高周波標準偏差で並べ、下位 flat_pct% を採る。
    data/score_raw.py のノイズ床推定と同じ考え方。
    """
    hp = _highpass(gray)
    ys, xs = np.nonzero(ann)
    if len(ys) < patch * patch:
        return 0.0
    stds = []
    step = max(patch // 2, 1)
    for y in range(ys.min(), max(ys.max() - patch, ys.min()) + 1, step):
        for x in range(xs.min(), max(xs.max() - patch, xs.min()) + 1, step):
            sub = ann[y:y + patch, x:x + patch]
            if sub.shape != (patch, patch) or sub.mean() < 0.9:
                continue
            stds.append(float(hp[y:y + patch, x:x + patch].std()))
    if not stds:
        return float(hp[ann].std())
    return float(np.percentile(stds, flat_pct))


def inject_grain(img: np.ndarray, mask: np.ndarray, ann: np.ndarray,
                 rng: np.random.Generator | None = None,
                 strength: float = 1.0) -> np.ndarray:
    """マスク内に、周囲と統計の一致する粒状を注入する。

    **振幅は高周波帯で合わせる。** パッチ全体の分散から σ を取ると、
    平坦なパッチを選んでもシーンの低周波成分が残って過大評価になり、
    粒状を入れすぎて逆に継ぎ目が目立つ (実測で継ぎ目 0.88 -> 1.31 に悪化した)。
    ノイズと区別できるのは高周波側だけなので、
    合成した粒状と周囲の実測値を**同じ高周波フィルタを通して**突き合わせる。
    """
    rng = rng or np.random.default_rng()
    psd, _ = estimate_noise_psd(img, ann)
    if ann.sum() < 64:
        return img

    gray = img[..., :3].mean(axis=2) if img.ndim == 3 else img
    target = estimate_grain_sigma(gray, ann)
    if target <= 1e-9:
        return img

    out = img.copy()
    h, w = mask.shape
    for c in range(img.shape[2]):
        g = synthesize_grain((h, w), psd, 1.0, rng)
        s = float(_highpass(g).std())
        if s <= 1e-12:
            continue
        out[..., c][mask] += (g * (target * strength / s))[mask]
    return out


# --------------------------------------------------------------------------
# 統合
# --------------------------------------------------------------------------

def harmonize(fill: np.ndarray, img: np.ndarray, mask: np.ndarray,
              valid: np.ndarray | None = None,
              rng: np.random.Generator | None = None,
              do_color: bool = True, do_poisson: bool = True,
              do_grain: bool = True) -> np.ndarray:
    """Stage D の全体。知覚空間 (H, W, C) float32 で受けて返す。

    fill : Stage B/C の合成結果。**マスクの外側にも最低 ANNULUS 幅まで伸びていること。**
           勾配ドメイン合成は「境界での元画像との食い違い」を境界条件に使うため、
           マスク外を先に元画像へ戻した fill を渡すと食い違いが恒等的に 0 になり、
           調和膜が消えて**何も起きない**。Stage B/C は膨張領域を処理するので、
           その出力をそのまま渡せばよい。
    img  : 元画像
    mask : 埋める領域 (bool)
    valid: 使ってよい画素 (飽和マップの否定)。None なら全画素
    """
    mask = mask.astype(bool)
    ann = annulus_of(mask, ANNULUS, valid)

    out = fill
    if do_color:
        out = match_color(out, img, mask, ann)
    if do_poisson:
        out = poisson_blend(out, img, mask)
    # マスク外は必ず元画像を保つ
    out = np.where(mask[..., None], out, img)
    if do_grain:
        out = inject_grain(out, mask, ann, rng)
        out = np.where(mask[..., None], out, img)
    return out.astype(np.float32)
