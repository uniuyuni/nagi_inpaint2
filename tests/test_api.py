"""ShadeWave 契約 (`helpers/ri_helper.py` の形) の検査。

**モデルを読まない。** ここで守るのは契約と幾何であって品質ではない。
特に大事なのは **bbox を尊めること** — 渡されるマスクは全箇所を含むので、
無視すると離れた 2 箇所を塗ったときに画面全体が 1 クロップになり、
実測で原寸に 12 対 1 で負ける L4 経路へ落ちる。
"""
import zlib

import numpy as np
import pytest

from nagi_inpaint2 import api
from nagi_inpaint2 import pipeline as P
from nagi_inpaint2.perceptual_space import from_perceptual, to_perceptual


class _Stub:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, sub, sm):
        return self.rng.random(sub.shape, dtype=np.float32)


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(P, "build_remover",
                        lambda name, device="mps", **kw:
                        _Stub(zlib.crc32(name.encode())))
    return P.MultiInpainter(cores=("lama", "yoeo"), device="cpu", poisson=False)


def _lin(h, w, seed=0):
    """**リニア HDR**。1.0 を超える画素を必ず含める (契約が許している)。"""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.05, 1.3, h, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(
        (y + rng.normal(0, 0.01, (h, w, 3))).astype(np.float32))


def test_知覚空間の往復が厳密(pipe):
    x = _lin(64, 64)
    p, w = to_perceptual(x)
    assert np.abs(from_perceptual(p, w) - x).max() < 1e-5


def test_bbox_の外は書き換えない(pipe):
    img = _lin(1400, 1800)
    before = img.copy()
    mask = np.zeros(img.shape[:2], np.float32)
    mask[600:760, 800:900] = 1.0
    bbox = (800, 600, 100, 160)          # (x, y, w, h)
    out = api.predict_helper(pipe, img, mask, bbox)
    assert out is img, "in-place で同じ配列を返す契約"
    m = mask > 0.5
    assert not np.allclose(img[m], before[m]), "穴が埋まっていない"


def test_離れた2箇所でも原寸のまま処理する(pipe):
    """**分割は呼び出し側 (get_multiple_mask_bbox) が済ませている。**
    bbox を無視して全体の外接矩形を取ると、クロップが画面全体になる。"""
    H, W = 3000, 4000
    img = _lin(H, W)
    mask = np.zeros((H, W), np.float32)
    mask[300:460, 400:500] = 1.0         # 左上
    mask[2500:2660, 3500:3600] = 1.0     # 右下
    for bbox in ((400, 300, 100, 160), (3500, 2500, 100, 160)):
        y0, x0, side = api._region(img.shape, bbox, pipe.crop)
        assert side == pipe.crop, f"クロップが広がっている: {side}"
        cands = api.predict_candidates(pipe, img, mask, bbox)
        for _, (ay, ax, q) in cands.items():
            assert q.shape[0] <= pipe.crop and q.shape[1] <= pipe.crop


def test_他の塗り箇所を巻き込まない(pipe):
    """同じクロップの中に別の箇所があっても、**この呼び出しは bbox の分だけ消す。**"""
    img = _lin(1000, 1000)
    before = img.copy()
    mask = np.zeros(img.shape[:2], np.float32)
    mask[480:540, 480:520] = 1.0         # 担当する箇所
    mask[480:540, 600:640] = 1.0         # 同じクロップに入る別の箇所
    api.predict_helper(pipe, img, mask, (480, 480, 40, 60))
    other = np.zeros(img.shape[:2], bool)
    other[480:540, 600:640] = True
    assert (img[other] == before[other]).all(), "担当外の箇所まで消している"


def test_候補は2つ返り位置が揃う(pipe):
    img = _lin(900, 900)
    mask = np.zeros(img.shape[:2], np.float32)
    mask[400:500, 400:460] = 1.0
    c = api.predict_candidates(pipe, img, mask, (400, 400, 60, 100))
    assert set(c) == {"lama", "yoeo"}
    (y1, x1, q1), (y2, x2, q2) = c["lama"], c["yoeo"]
    assert (y1, x1) == (y2, x2) and q1.shape == q2.shape
    assert not np.allclose(q1, q2), "2 案が同一なら並べる意味がない"
    assert not np.shares_memory(q1, img), "候補は画像を書き換えていないこと"


def test_候補を作っても画像は無傷(pipe):
    img = _lin(800, 800)
    before = img.copy()
    mask = np.zeros(img.shape[:2], np.float32)
    mask[350:430, 350:400] = 1.0
    api.predict_candidates(pipe, img, mask, (350, 350, 50, 80))
    assert (img == before).all()


@pytest.mark.parametrize("shape", [(1, "hw"), (3, "hw1")])
def test_マスクは_hw_でも_hw1_でも良い(pipe, shape):
    ch, kind = shape
    img = _lin(700, 700)
    m = np.zeros((700, 700), np.float32)
    m[300:360, 300:340] = 1.0
    mask = m if kind == "hw" else m[..., None]
    out = api.predict_helper(pipe, img.copy(), mask, (300, 300, 40, 60))
    assert out.shape == img.shape


def test_書き込めない配列は弾く(pipe):
    img = _lin(600, 600)
    img.flags.writeable = False
    mask = np.zeros(img.shape[:2], np.float32)
    mask[300:340, 300:330] = 1.0
    with pytest.raises(ValueError):
        api.predict_helper(pipe, img, mask, (300, 300, 30, 40))


def test_マスクが空なら無変更(pipe):
    img = _lin(600, 600)
    before = img.copy()
    out = api.predict_helper(pipe, img, np.zeros(img.shape[:2], np.float32),
                             (100, 100, 20, 20))
    assert (out == before).all()


def test_マスクの形が違えば弾く(pipe):
    img = _lin(600, 600)
    with pytest.raises(ValueError):
        api.predict_helper(pipe, img, np.zeros((500, 500), np.float32),
                           (100, 100, 20, 20))


def test_マスク外はリニアでもビット一致(pipe):
    """**知覚空間の往復では float32 で 2.4e-07 の誤差が出る** (2026-08-30 に検出)。
    パイプライン内部がビット一致を守っていても、変換結果をそのまま貼ると
    この層で SPEC §9.1 が崩れる。`api` はリニアの原画素を戻して守る。"""
    img = _lin(1000, 1000)
    before = img.copy()
    mask = np.zeros(img.shape[:2], np.float32)
    mask[450:530, 460:510] = 1.0
    api.predict_helper(pipe, img, mask, (460, 450, 50, 80))
    outside = mask <= 0.5
    assert (img[outside] == before[outside]).all(), \
        "マスク外が 1 ビットでも変わっている"
