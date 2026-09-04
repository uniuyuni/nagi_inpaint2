"""知覚空間の往復変換 (SPEC §3.3)。

作業色空間は **リニア ProPhoto RGB** (§6.0 の現像設定に一致)。
リニアのまま L1 を取ると損失が高輝度領域に集中して破綻するため、
`radiance_denoise` と同型の知覚空間に写して処理し、最後に逆変換する。

    white = percentile(mean(x[..., :3]), 99.5)
    p     = sign(x) · (|x| / white)^(1/γ)          γ = 2.2
    x     = sign(p) · |p|^γ · white                 (厳密に可逆)

**`radiance_denoise` の `max(x, 0)` ではなく符号付きべき乗を使う。**
ProPhoto RGB は広色域のため、カメラ行列変換後に色域外の負値が正常に発生する
(実測: X-E3 サンプルで min=-0.0284、全画素の 0.0015%)。
負値をクリップすると彩度の高い被写体の境界で色が飛ぶ。

**`white` は画像全体で一度だけ決める。** タイルごとに再計算すると
タイル境界に不連続が出る。
"""
from __future__ import annotations

import numpy as np

GAMMA = 2.2
WHITE_PCT = 99.5

# data/extract.py の shard と同じ量子化定数。読み出し側は json の値を優先すること。
Q_SCALE = 12000
Q_OFFSET = 8000


def estimate_white(x: np.ndarray, pct: float = WHITE_PCT) -> float:
    """リニア画像から white を決める。画像全体で一度だけ呼ぶこと。"""
    w = float(np.percentile(x[..., :3].mean(axis=-1), pct))
    return w if np.isfinite(w) and w > 0 else 1.0


def to_perceptual(x, white: float, gamma: float = GAMMA):
    """リニア -> 知覚空間。numpy / torch のどちらでも動く。"""
    if isinstance(x, np.ndarray):
        return np.sign(x) * np.power(np.abs(x) / white, 1.0 / gamma)
    return x.sign() * (x.abs() / white).pow(1.0 / gamma)


def to_linear(p, white: float, gamma: float = GAMMA):
    """知覚空間 -> リニア。to_perceptual の厳密な逆。"""
    if isinstance(p, np.ndarray):
        return np.sign(p) * np.power(np.abs(p), gamma) * white
    return p.sign() * p.abs().pow(gamma) * white


def quantize(p: np.ndarray, scale: int = Q_SCALE, offset: int = Q_OFFSET) -> np.ndarray:
    """知覚空間 -> uint16。符号付きなのでオフセット表現で負値を保持する。"""
    return np.clip(np.rint(p * scale) + offset, 0, 65535).astype(np.uint16)


def dequantize(u: np.ndarray, scale: int = Q_SCALE, offset: int = Q_OFFSET) -> np.ndarray:
    """uint16 -> 知覚空間 (float32)。"""
    return (u.astype(np.float32) - offset) / scale


def byte_shuffle(arr: np.ndarray) -> bytes:
    """uint16 の上位/下位バイトを分離する。zstd の圧縮率が 1.15x -> 1.50x になる。

    16bit 写真データの下位ビットはノイズそのもので原理的に圧縮できないが、
    バイトを分離すると上位バイト側が滑らかになり、そちらだけがよく縮む。
    """
    a = arr.view(np.uint8).reshape(-1, 2)
    return np.concatenate([a[:, 0], a[:, 1]]).tobytes()


def byte_unshuffle(buf: bytes, shape, dtype=np.uint16) -> np.ndarray:
    """byte_shuffle の逆。reshape だけで済むので学習ループ内でも軽い。"""
    a = np.frombuffer(buf, dtype=np.uint8)
    half = len(a) // 2
    out = np.empty((half, 2), dtype=np.uint8)
    out[:, 0], out[:, 1] = a[:half], a[half:]
    return out.view(dtype).reshape(shape)
