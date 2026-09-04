"""リニア <-> 知覚空間の変換 (SPEC §3.3)。**torch も libraw も import しない。**

**なぜ独立したモジュールなのか**

この規約は元々 `data/extract.py` にあったが、あのファイルは先頭で
`libraw_enhanced` を import する。**libraw_enhanced と torch は両方 libomp を
リンクしており、同一プロセスに載せると衝突する** (本リポジトリで何度も踏んでいる)。
推論側 (`nagi_inpaint2/api.py`) は torch を使うので、
`data/extract.py` からは定数を借りられない。

かといって数値を 2 か所に書くと**片方だけずれても誰も気づかない**。
そこで**どちらからも import できる、依存の無いモジュール**に置く。
`data/extract.py` はここから import し、`GAMMA` / `WHITE_PCT` の名前を
再輸出するので `from extract import GAMMA, WHITE_PCT` は従来どおり動く。

**規約**

    p = sign(x) * (|x| / white) ** (1/GAMMA)
    x = sign(p) * |p| ** GAMMA * white

`white` は**画像全体で一度だけ決める**。クロップごとに決めると
shard の作り方とずれる (`data/build_removal_bench.py` の注記と同じ理由)。
"""
from __future__ import annotations

import numpy as np

GAMMA = 2.2
WHITE_PCT = 99.5


def white_point(img: np.ndarray, stride: int = 1) -> float:
    """白基準。`stride > 1` なら間引いて求める。

    **間引いてよい根拠 (実測、40MP)**: 全画素 0.42 秒に対し 8 画素おきは 0.005 秒で
    **84 倍速く、値の差は 0.053%**。さらに除去器は内部で自動露出を掛けて
    線形に測り直すので、この程度の差は結果に出ない。
    40MP でも 8 画素おきで 62 万標本あり、99.5 パーセンタイルには十分である。
    """
    g = img[..., :3].mean(axis=2)
    if stride > 1:
        g = g[::stride, ::stride]
    w = float(np.percentile(g, WHITE_PCT))
    return w if (np.isfinite(w) and w > 0) else 1.0


def to_perceptual(img: np.ndarray,
                  white: float | None = None) -> tuple[np.ndarray, float]:
    """リニア -> 知覚空間。`(p, white)` を返す。負値も符号を保って通す。"""
    if white is None:
        white = white_point(img)
    p = np.sign(img) * np.power(np.abs(img) / white, 1.0 / GAMMA)
    return p.astype(np.float32), float(white)


def from_perceptual(p: np.ndarray, white: float) -> np.ndarray:
    """知覚空間 -> リニア。`to_perceptual` の厳密な逆写像。"""
    return (np.sign(p) * np.power(np.abs(p), GAMMA) * white).astype(np.float32)
