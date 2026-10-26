"""重みの一覧・取得・置き場所の解決 (ゴール: 「正しい重みが、指定した場所から読まれる」)。"""
import os

import pytest

from nagi_inpaint2 import assets, paths


def test_一覧は検証済み構成の重みだけで_UNet_や_fp16_重複を含まない():
    names = {a.path for a in assets.ASSETS}
    assert "yoeo/yoeo.pth" in names and "big-lama.pt" in names
    assert not any("unet/" in n and not n.endswith("config.json") for n in names), \
        "UNet の重みは YOEO が置き換えるので取らない"
    assert not any("fp16" in n for n in names), "fp16 版に替えると実測の前提が崩れる"
    for a in assets.ASSETS:
        assert len(a.sha256) == 64 and a.size > 0
        if a.hf:
            assert len(a.hf[1]) == 40, f"{a.path}: リビジョンを固定すること"


def test_check_は大きさ違いと欠落を検出する(tmp_path):
    a = assets.ASSETS[2]                       # 小さい設定ファイル
    f = tmp_path / a.path
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x" * a.size)               # 大きさだけ合っている偽物
    bad = {b.path for b in assets.check(str(tmp_path))}
    assert a.path not in bad, "浅い検査は大きさだけを見る"
    assert a.path in {b.path for b in assets.check(str(tmp_path), deep=True)}, \
        "深い検査は中身の違いを見逃さない"
    assert len(bad) == len(assets.ASSETS) - 1


def test_流用元が壊れていたら置かずに失敗を返す(tmp_path):
    a = assets.ASSETS[2]
    mirror = tmp_path / "mirror"
    (mirror / a.path).parent.mkdir(parents=True)
    (mirror / a.path).write_bytes(b"x" * a.size)
    dest = tmp_path / "dest"
    # 他は取りに行かせない (ネットワークを使わない)
    orig = assets.ASSETS
    try:
        assets.ASSETS = (a,)
        failed = assets.fetch(str(dest), mirror=str(mirror), log=lambda *_: None)
    finally:
        assets.ASSETS = orig
    assert [f.path for f in failed] == [a.path]
    assert not (dest / a.path).exists(), "検証に落ちたファイルを残さない"


def test_明示した重みの場所が作業ディレクトリより優先される(tmp_path, monkeypatch):
    """作業ディレクトリに models/ があっても、$NAGI_INPAINT2_MODELS が勝つこと。"""
    cwd = tmp_path / "cwd"
    (cwd / "models").mkdir(parents=True)
    (cwd / "models" / "big-lama.pt").write_bytes(b"cwd")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv(paths.ENV, str(tmp_path / "explicit"))
    assert paths.resolve("models/big-lama.pt") == str(tmp_path / "explicit" / "big-lama.pt")


def test_明示した場所に無くても他を探さない(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(paths.ENV, str(tmp_path / "empty"))
    r = paths.resolve("models/yoeo/yoeo.pth")
    assert r.startswith(str(tmp_path / "empty")) and not os.path.exists(r)
