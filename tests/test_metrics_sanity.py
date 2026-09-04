"""全指標の陰性対照テスト。device は "cpu" 固定。

**なぜこれが要る (背景)**

このセッションで**同じ種類の失敗を4回**繰り返した。いずれも「学習中に見ていた
数値が、最終評価の数値と乖離していた」もの:

  - BatchNorm の running_var を「パラメータが動いている」と誤読した
  - 照合コスト学習で cache-val の regret は 1.481→1.236 と改善したのに、
    下流の PSNR は -2.1 dB 悪化した
  - **直近**: WTA 学習の検証が `grain 1.021` (理想 1.0) を報告したが、
    人物マスクで測ると **0.098** だった。学習側の検証が**混合マスク (電線など
    細いもの)** で測っていたため。細い穴では grain 比が自動的に 1.0 付近になり、
    **ぼけを検出できない**。

**指標が壊れを検出できなければ、その指標を使った判断は全て無意味になる。**
このファイルは `eval/metrics.py` の `grain_ratio` / `psd_distance` /
`boundary_discontinuity` と `eval/perceptual.py` の `Perceptual` が、
「壊れた補完」を実際に壊れていると判定できることを確認する。

**test_grain_is_blind_on_thin_masks について (実測で分かったこと)**

`grain_ratio` は境界の勾配を拾わないよう `inner = mask & ~dilate(~mask, 3)`
(3px 侵食) でマスクを内側に縮めてから測る。侵食半径が固定 3px であるため、
**幅が概ね 7px 未満のマスクでは侵食後の内部が空になり、常に `nan` を返す**
(電線マスクの太さは `data/masks.py` の `wire()` で 1〜5px — 常にこの範囲に
入る。実際に `masks.wire()` で生成したマスクで確認しても、ぼかしの有無に
関わらず全て `nan` だった)。

この `nan` は「測れない」で終わらない。`train/train_stage_fw.py` の
`validate()` は

    if np.isfinite(g):
        tot_grain += g
        n_grain += 1

というふうに **`nan` を黙って平均から除外する**。つまり細い (電線的な) マスク
は検証バッチに混ざっていても一切カウントされず、平均は「たまたま測れた
(=侵食で潰れない) マスク」だけで計算される。これが「WTA 学習の検証が
grain 1.021 を報告したが、人物マスクでは 0.098 だった」の実際の機構であり、
`grain_ratio` 自身は「1.0 付近の値を返す」のではなく「何も返せず、それが
黙って握り潰される」という、より始末の悪い盲点を持っている。

このテストは「幅 2px の帯マスクでは、ぼかしの前後で `grain_ratio` の結果が
(nan であれ、有限値であれ) 1.0 から明確に離れない」ことを確認する — つまり
`grain_ratio` がこのマスク形状に対してぼけを検出する能力を一切持たないこと
を、対になる人物サイズマスク (60x80px、確実に検出できる) と対比して示す。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import boundary_discontinuity, grain_ratio, psd_distance  # noqa: E402
from eval.perceptual import Perceptual                                      # noqa: E402

DEVICE = "cpu"
N = 256


def _photo_like(n: int, seed: int) -> np.ndarray:
    """自然画像に近い、細かいディテールと構造を両方持つ合成テクスチャ。

    `tests/test_perceptual.py` の同名ヘルパーと同じ考え方 (低周波構造 + 微小
    ノイズ)。値域はこのリポジトリの慣習 (知覚空間, おおよそ 0..1.2) に合わせる。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    base = (0.5
            + 0.2 * np.sin(xx / 5.0)
            + 0.2 * np.cos(yy / 7.0)
            + 0.1 * np.sin((xx + yy) / 3.0))
    img = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.2).astype(np.float32)


def _blur_full(x: np.ndarray, sigma: float) -> np.ndarray:
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        out[..., c] = gaussian_filter(x[..., c], sigma=sigma)
    return out


def _blob_mask(n: int = N) -> np.ndarray:
    """人物サイズの塊マスク (60x80px)。画像中央に置き、境界/アニュラス処理
    (侵食3px、アニュラス最大32px) に十分な余白を確保する。
    """
    m = np.zeros((n, n), dtype=bool)
    cy, cx = n // 2, n // 2
    m[cy - 40:cy + 40, cx - 30:cx + 30] = True   # 80(H) x 60(W)
    return m


def _thin_mask(n: int = N, width: int = 2) -> np.ndarray:
    """幅 `width` px の帯マスク (電線を模す)。画面を縦断する直線状。"""
    m = np.zeros((n, n), dtype=bool)
    c0 = n // 2 - width // 2
    m[40:n - 40, c0:c0 + width] = True
    return m


@pytest.fixture(scope="module")
def perceptual() -> Perceptual:
    return Perceptual(device=DEVICE)


# --- 1. ぼかし: grain は 1.0 から下に、psd は増え、Perceptual も増える ------

def test_blur_breaks_completion(perceptual):
    """マスク領域を強くぼかした補完を、GT (未変更部分と統計的に同じ) と比べる。"""
    x = _photo_like(N, seed=0)
    mask = _blob_mask()
    blurred_full = _blur_full(x, sigma=3.0)
    out = x.copy()
    out[mask] = blurred_full[mask]

    g_gt = grain_ratio(x, mask)          # 未変更 (=完璧な補完) の基準値、1.0 近傍
    g_blur = grain_ratio(out, mask)
    print(f"\n[blur] grain: gt={g_gt:.4f}  blurred={g_blur:.4f}")
    assert np.isfinite(g_gt) and 0.8 <= g_gt <= 1.2
    assert g_blur < 0.8, "ぼかしで grain が 1.0 から下に離れるはず"

    d_gt = psd_distance(x, mask)
    d_blur = psd_distance(out, mask)
    print(f"[blur] psd: gt={d_gt:.4f}  blurred={d_blur:.4f}")
    assert d_blur > d_gt, "ぼかしで psd (距離) が増えるはず"

    p_gt = perceptual(x, x, mask=mask)
    p_blur = perceptual(out, x, mask=mask)
    print(f"[blur] perceptual: gt={p_gt:.6f}  blurred={p_blur:.6f}")
    assert p_blur > p_gt, "ぼかしで知覚距離が増えるはず"


# --- 2. ノイズ: grain は 1.0 から上に離れる -----------------------------------

def test_noise_breaks_completion():
    x = _photo_like(N, seed=1)
    mask = _blob_mask()
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 0.15, x.shape).astype(np.float32)
    out = x.copy()
    out[mask] = np.clip(x[mask] + noise[mask], 0.0, 1.2)

    g_gt = grain_ratio(x, mask)
    g_noise = grain_ratio(out, mask)
    print(f"\n[noise] grain: gt={g_gt:.4f}  noisy={g_noise:.4f}")
    assert g_noise > 1.5, "ノイズ付加で grain が 1.0 から上に離れるはず"


# --- 3. ずらし: boundary が増える ---------------------------------------------

def test_shift_breaks_completion():
    x = _photo_like(N, seed=2)
    mask = _blob_mask()
    shifted = np.roll(x, shift=(40, 40), axis=(0, 1))
    out = x.copy()
    out[mask] = shifted[mask]

    b_gt = boundary_discontinuity(x, mask)
    b_shift = boundary_discontinuity(out, mask)
    print(f"\n[shift] boundary: gt={b_gt:.4f}  shifted={b_shift:.4f}")
    assert b_shift > b_gt
    assert b_shift > 2.0, "大きくずらした継ぎ目は boundary が明確に増えるはず"


# --- 4. 記録: grain_ratio は細い帯マスクに対して盲目 (今回の失敗の再現) -------

def test_grain_is_blind_on_thin_masks():
    """幅 2px の細い帯マスクでは、マスク内をどれだけぼかしても `grain_ratio` が
    1.0 付近から動かない (実測では、侵食で内部が空になり `nan` を返し続ける)
    ことを明示的に確認する。

    この盲点のせいで WTA 学習の検証が grain 1.021 を報告したが、
    人物マスクでは 0.098 だった。

    実測 (このテスト内、`_thin_mask(width=2)` に sigma=3.0 のぼかしを適用):
    ぼかし前後どちらも `grain_ratio` は `nan`
    (`inner = mask & ~dilate(~mask, 3)` の侵食半径が固定 3px なので、
    幅 7px 未満のマスクは侵食後の内部が空になる。`data/masks.py` の
    `wire()` が作る実際の電線マスク (太さ1〜5px) でも同様に確認済み)。
    `nan` は「1.0 に近い」以上に始末が悪い — `train/train_stage_fw.py` の
    `validate()` は `if np.isfinite(g):` で黙って平均から除外するため、
    細いマスクは検証集計に一切寄与しない。

    対比として、同じぼかしを人物サイズの塊マスク (60x80px) に適用すると
    grain が明確に 0.5 未満へ落ちる (実測 ~0.15) ことを併せて確認する。
    **この対比がテストの主眼**: 同じ「壊れ」を与えても、マスク形状だけで
    指標の感度が全く違う。
    """
    x = _photo_like(N, seed=42)
    blurred_full = _blur_full(x, sigma=3.0)

    thin_mask = _thin_mask(width=2)
    thin_out = x.copy()
    thin_out[thin_mask] = blurred_full[thin_mask]
    g_thin_gt = grain_ratio(x, thin_mask)
    g_thin_blur = grain_ratio(thin_out, thin_mask)
    print(f"\n[thin 2px] grain: gt={g_thin_gt}  blurred={g_thin_blur}")
    # nan であれ有限値であれ、1.0 から明確に離れてはいけない (=検出できない)
    for g in (g_thin_gt, g_thin_blur):
        assert np.isnan(g) or 0.8 <= g <= 1.2

    blob_mask = _blob_mask()
    blob_out = x.copy()
    blob_out[blob_mask] = blurred_full[blob_mask]
    g_blob_gt = grain_ratio(x, blob_mask)
    g_blob_blur = grain_ratio(blob_out, blob_mask)
    print(f"[blob 60x80] grain: gt={g_blob_gt:.4f}  blurred={g_blob_blur:.4f}")
    assert np.isfinite(g_blob_gt) and 0.8 <= g_blob_gt <= 1.2
    assert g_blob_blur < 0.5, "人物サイズマスクなら同じぼかしを明確に検出できる"
