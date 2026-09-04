"""実物体シルエット種別 (`data.masks.silhouette`) の回帰テスト。

device 不要、CPU のみ。乱数のみで完結する。

このモジュールを追加した理由: 手続き生成の `blob` は極座標で揺らした
閉曲線にすぎず、常に凸に近い一塊の形にしかならない。人物 (本用途の主対象)
の輪郭が持つ頭部+肩のくびれ・腕や脚の突起・局所的な凹みを再現できない。
`silhouette` は `data/masks_dict.py` が MobileSAM で実タイルから抽出した
実シルエット (`cache/mask_dict.npz`) を貼ることでこれを補う。テスト4で
その狙いどおり形状統計 (充填率) が blob と有意に異なることを確認する。
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import masks  # noqa: E402


def _fill_rate(m: np.ndarray) -> float:
    """外接矩形に対する充填率 (面積 / 外接矩形面積)。"""
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return 0.0
    bh = int(ys.max() - ys.min() + 1)
    bw = int(xs.max() - xs.min() + 1)
    return float(m.sum()) / float(bh * bw)


def test_registered_in_generators():
    """1. silhouette が GENERATORS に登録されている。"""
    assert "silhouette" in masks.GENERATORS
    assert masks.GENERATORS["silhouette"] is masks.silhouette


def test_sample_mask_shape_dtype():
    """2. sample_mask(512, rng, kind="silhouette") が (512,512) の bool 配列を返す。"""
    rng = np.random.default_rng(0)
    m, meta = masks.sample_mask(512, rng, kind="silhouette")
    assert m.shape == (512, 512)
    assert m.dtype == bool
    assert meta["kind"] == "silhouette"


def test_area_matches_target_across_seeds():
    """3. target_area=0.01 のとき、実面積が 0.003〜0.03 に収まる (複数シードで確認)。

    sample_mask は面積が目標に収まるまで数回引き直す (既定 max_tries=6) ので、
    その仕組みが実際に機能していることを複数シードで確かめる。
    """
    n_seeds = 30
    failures = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        m, meta = masks.sample_mask(512, rng, kind="silhouette", target_area=0.01)
        area = float(m.mean())
        if not (0.003 <= area <= 0.03):
            failures.append((seed, area))
    assert not failures, f"面積が範囲外だったシード: {failures}"


def test_fill_rate_differs_from_blob():
    """4. 充填率 (面積 / 外接矩形面積) の分布が blob と有意に異なる。

    silhouette 種別を追加する理由そのものなので必ず確認する。blob は極座標で
    緩く揺らした閉曲線で凸に近い塊にしかならず充填率が高く出やすい。一方
    silhouette は頭部+肩のくびれや四肢の突起を含むため外接矩形に対して
    隙間が多く、充填率が低くなる。中央値の差が 0.05 以上あることを確認する。
    """
    rng = np.random.default_rng(42)
    blob_fr, sil_fr = [], []
    for _ in range(300):
        m, _ = masks.sample_mask(256, rng, kind="blob", target_area=0.02)
        blob_fr.append(_fill_rate(m))
        m, _ = masks.sample_mask(256, rng, kind="silhouette", target_area=0.02)
        sil_fr.append(_fill_rate(m))

    blob_med = float(np.median(blob_fr))
    sil_med = float(np.median(sil_fr))
    assert blob_med - sil_med >= 0.05, (
        f"blob 中央値={blob_med:.4f} silhouette 中央値={sil_med:.4f} "
        f"差={blob_med - sil_med:.4f} (0.05 以上を期待)"
    )


def test_import_succeeds_without_mask_dict(monkeypatch, tmp_path):
    """5. cache/mask_dict.npz が無い環境でも import data.masks が成功する。

    silhouette を呼ばない限りエラーにならないこと、他の種別は影響を受け
    ないことを確認する。npz が存在しない cwd に切り替えてモジュールを
    再読み込みすることで「ファイルが無い環境」を再現する。
    """
    monkeypatch.chdir(tmp_path)  # cache/mask_dict.npz が存在しない cwd
    try:
        importlib.reload(masks)  # import 自体は例外を出さないはず

        rng = np.random.default_rng(0)
        for kind in ("wire", "dust", "stroke", "rect", "blob"):
            m, _ = masks.sample_mask(64, rng, kind=kind)
            assert m.shape == (64, 64)

        with pytest.raises(FileNotFoundError):
            masks.silhouette(64, rng)
    finally:
        importlib.reload(masks)  # 実ファイルのある状態に戻す
