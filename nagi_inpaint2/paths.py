"""重みの置き場所を解決する。**呼び出し元の作業ディレクトリに依存させない。**

重みのパスは `"models/big-lama.pt"` のような**相対パス**で書かれている。
`nagi_inpaint2` のリポジトリ直下で動かす分には問題ないが、
**ShadeWave から呼ぶと作業ディレクトリが向こうになるので全部見つからなくなる**
(2026-08-30 に統合作業で判明)。

そこで相対パスは**このパッケージの親 (リポジトリ直下)** を基準に解決する。
環境変数 `NAGI_INPAINT2_MODELS` があればそちらを優先するので、
重みを別の場所 (ShadeWave の `checkpoints/` など) に置くこともできる。
絶対パスはそのまま通す。
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = "NAGI_INPAINT2_MODELS"


def models_dir() -> str:
    return os.environ.get(ENV) or os.path.join(ROOT, "models")


def resolve(path: str) -> str:
    """相対パスをリポジトリ (または `$NAGI_INPAINT2_MODELS`) 基準にする。

    **既にある場所を壊さない**: 作業ディレクトリからの相対で実在するなら、
    そちらを優先する (リポジトリ内で動かす従来の使い方をそのまま維持)。
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    if path.startswith("models" + os.sep) or path.startswith("models/"):
        cand = os.path.join(models_dir(), path.split("/", 1)[1]
                            if "/" in path else path[len("models"):].lstrip(os.sep))
        if os.path.exists(cand):
            return cand
    cand = os.path.join(ROOT, path)
    return cand if os.path.exists(cand) else path
