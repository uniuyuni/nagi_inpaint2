"""`MultiInpainter` の不変条件 (SPEC §9.1)。

**モデルを読まずに測る。** ここで守りたいのは品質ではなく
「マスク外を絶対に触らない」「画像全体を複製しない」という構造の性質なので、
除去コアは差し替え可能な偽物でよい。品質は `eval/accept.py` で測っている。
"""
import zlib

import numpy as np
import pytest

from nagi_inpaint2 import pipeline as P


class _Stub:
    """クロップと同じ形の乱数を返すだけの偽コア。**穴の外も塗りつぶす** ——
    合成側が守っていなければテストが落ちるようにするため。"""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, sub, sm):
        return self.rng.random(sub.shape, dtype=np.float32)


@pytest.fixture
def multi(monkeypatch):
    # `hash()` は実行ごとに変わる (PYTHONHASHSEED) ので使わない。
    # 種が偶然一致すると「2 案が別物か」のテストが不安定に落ちる。
    monkeypatch.setattr(P, "build_remover",
                        lambda name, device="mps", **kw:
                        _Stub(zlib.crc32(name.encode())))
    return P.MultiInpainter(cores=("a", "b"), device="cpu", poisson=False)


def _img(h, w, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    y = np.linspace(0.2, 0.7, h, dtype=np.float32)[:, None, None]
    return (y + rng.normal(0, 0.02, (h, w, 3)).astype(np.float32)).astype(dtype)


def test_マスク外はビット単位で原画素(multi):
    img = _img(1600, 2400)
    mask = np.zeros(img.shape[:2], bool)
    mask[700:900, 1100:1250] = True
    y0, x0, pats = multi.patches(img, mask)
    sub = img[y0:y0 + 512, x0:x0 + 512]
    sm = mask[y0:y0 + 512, x0:x0 + 512]
    assert set(pats) == {"a", "b"}
    for name, q in pats.items():
        assert q.shape == sub.shape
        assert (q[~sm] == sub[~sm]).all(), f"{name} がマスク外を書き換えた"
        assert np.abs(q[sm] - sub[sm]).mean() > 1e-6, f"{name} が穴を埋めていない"


def test_候補どうしは別物(multi):
    img = _img(1024, 1024)
    mask = np.zeros(img.shape[:2], bool)
    mask[400:600, 400:520] = True
    _, _, pats = multi.patches(img, mask)
    sm_area = mask[400:600, 400:520]
    assert not np.allclose(pats["a"], pats["b"]), "2 案が同一なら並べる意味がない"
    assert sm_area.any()


def test_入力を書き換えない(multi):
    img = _img(900, 900)
    before = img.copy()
    mask = np.zeros(img.shape[:2], bool)
    mask[400:500, 400:460] = True
    multi.patches(img, mask)
    assert (img == before).all(), "入力画像が破壊されている"


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_入力の型に依らない(multi, dtype):
    """`float16` の巨大画像を渡しても、**全体を float32 に複製してはいけない**。
    複製すると 100MP で 1.26GB 増える (2026-08-30 に実測して是正)。"""
    img = _img(1200, 1200, dtype=dtype)
    mask = np.zeros(img.shape[:2], bool)
    mask[500:700, 500:600] = True
    y0, x0, pats = multi.patches(img, mask)
    assert img.dtype == dtype, "入力の型が変わっている = 全体を複製している"
    for q in pats.values():
        assert q.dtype == np.float32


def test_マスクが画像の端にあっても範囲に収まる(multi):
    img = _img(700, 700)
    mask = np.zeros(img.shape[:2], bool)
    mask[:60, :60] = True
    y0, x0, pats = multi.patches(img, mask)
    q = pats["a"]
    assert y0 >= 0 and x0 >= 0
    assert y0 + q.shape[0] <= img.shape[0]
    assert x0 + q.shape[1] <= img.shape[1]


def test_画像より小さいマスクが無い場合(multi):
    img = _img(600, 600)
    mask = np.zeros(img.shape[:2], bool)
    out = multi(img, mask)
    for q in out.values():
        assert (q == img.astype(np.float32)).all(), "マスクが空なら無変更のはず"


def test_call_は貼り付けた全体を返す(multi):
    img = _img(800, 800)
    mask = np.zeros(img.shape[:2], bool)
    mask[300:420, 350:430] = True
    y0, x0, pats = multi.patches(img, mask)
    full = multi(img, mask)
    for name in ("a", "b"):
        f = full[name]
        assert f.shape == img.shape
        # 貼った矩形の外は原画素のまま
        m = np.zeros(img.shape[:2], bool)
        m[y0:y0 + pats[name].shape[0], x0:x0 + pats[name].shape[1]] = True
        assert (f[~m] == img.astype(np.float32)[~m]).all()
