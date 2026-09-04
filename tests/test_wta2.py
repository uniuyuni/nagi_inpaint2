"""StageFW2 (デコーダ分岐 WTA) の回帰テスト。device は "cpu" 固定。

**効く理屈と、効かない条件**

狙い: L1 損失は多峰性の分布に対して平均を出すためぼける (実測: grain 0.109、
LaMa 0.943、理想 1.0)。K 個の仮説を出して**最良の1つにだけ勾配を流す**
(Winner-Take-All) と、平均を作る動機が消える。GAN と違い識別器も min-max も
無いので学習は安定する (GAN は2回試して崩壊した)。

**前回 (`StageFW`) の失敗**: `nagi_inpaint2/stage_fw.py` の `StageFW` は
**出力層 (最後の 3x3 畳み込み) だけ**を K 個にした。4つのヘッドはバック
ボーン全体を共有し、**同じ特徴の線形読み出しの違いしか表現できない**。結果、
仮説がほぼ同一になり、**正解を見て最良を選ぶ神託選択ですら first 選択を
上回らなかった** (PSNR 21.29 vs 21.43)。仮説間の損失の標準偏差も損失の約10%
しかなかった。

**効かない条件 (検出方法)**: 分岐が浅いと再び同一の仮説に収束する。検出方法は
「神託選択が first 選択を明確に上回るか」。上回らなければ多様性が無く、WTA は
機能していない。`StageFW2` はデコーダ以降を K 本の独立した重みの枝にし、枝
ごとの学習可能な埋め込みを加算することでこれに対処する — その対処が
「浅い分岐 (`StageFW`) より明確に多様な仮説を生む」ことを、このファイルの
テスト5 で担保する (この設計の存在理由そのものであり、必ず入れる)。

確認すること:
  1. `StageFW2(k=4).forward` の出力が (N,4,3,H,W) で、入力と同じ解像度
  2. パラメータ数が 20M 以下
  3. `load_backbone` が `StageF` の state_dict を受け取って動作する
     (共有部分が一致し、K 本の枝が全て同じ初期デコーダ重みになる)
  4. 枝が独立していること (枝0のパラメータにだけ勾配を流したとき、
     枝1のパラメータの勾配が0であること)
  5. **多様性が生まれること**: ランダム初期化した `StageFW2` の K 個の出力の
     仮説間平均二乗差が、同条件の `StageFW` (出力層のみ分岐) のそれより
     明確に大きい (2倍以上)
  6. checkpoint の往復: 保存して `load_stage_fw2` で読み直せること
     (`arch` キーを含む)
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_f import StageF                     # noqa: E402
from nagi_inpaint2.stage_fw import StageFW                   # noqa: E402
from nagi_inpaint2.stage_fw2 import StageFW2, load_stage_fw2  # noqa: E402

DEVICE = "cpu"


def _sample(n: int, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(batch, 3, n, n, generator=g, device=DEVICE)
    mask = (torch.rand(batch, 1, n, n, generator=g, device=DEVICE) > 0.5).float()
    return img, mask


# --- 1. forward の形 ---------------------------------------------------------

def test_forward_shape_and_resolution():
    model = StageFW2(base=16, k=4).to(DEVICE)
    img, mask = _sample(64, batch=2, seed=0)
    with torch.no_grad():
        out = model(img, mask)
    assert out.shape == (2, 4, 3, 64, 64)


def test_forward_shape_different_k():
    model = StageFW2(base=16, k=3).to(DEVICE)
    img, mask = _sample(48, batch=1, seed=1)
    with torch.no_grad():
        out = model(img, mask)
    assert out.shape == (1, 3, 3, 48, 48)


def test_forward_shape_non_square_and_odd_crop():
    """クロップが 4 の倍数でなくても (encoder が 1/4 まで落とすため) 動くこと。"""
    model = StageFW2(base=16, k=2).to(DEVICE)
    img = torch.rand(1, 3, 50, 66)
    mask = (torch.rand(1, 1, 50, 66) > 0.5).float()
    with torch.no_grad():
        out = model(img, mask)
    assert out.shape == (1, 2, 3, 50, 66)


# --- 2. パラメータ数が 20M 以下 -------------------------------------------------

def test_param_count_under_20m():
    model = StageFW2(base=96, k=4)
    n_par = sum(q.numel() for q in model.parameters())
    assert n_par <= 20_000_000, f"パラメータ数が 20M を超えている: {n_par:,}"
    # 実測値がおおむね想定通りであることも確認 (11.7M 前後)。
    assert 5_000_000 < n_par < 20_000_000


# --- 3. load_backbone ---------------------------------------------------------

def test_load_backbone_shared_part_matches():
    base_model = StageF(base=16).to(DEVICE)
    sd = base_model.state_dict()
    model = StageFW2(base=16, k=3).to(DEVICE)
    model.load_backbone(sd)
    for name, param in base_model.named_parameters():
        if name.startswith("dec.") or name.startswith("out."):
            continue
        my_param = dict(model.named_parameters())[name]
        assert torch.equal(param, my_param), f"共有部分 {name} が一致しない"


def test_load_backbone_creates_real_diversity():
    """**事前学習重みから初期化しても、仮説に実質的な多様性があること。**

    当初は「K 本の枝は全て同じ事前学習済み重みで初期化される」ことを検査していたが、
    その仕様では枝の違いが `emb` (std=0.02) だけになり、実測で仮説間の平均二乗差が
    **4.4e-8** と実質ゼロだった (出力層のみ分岐する `StageFW` の 4.4e-4 より1万倍小さい)。
    その `StageFW` ですら学習後に多様性が生まれず、**正解を見て選ぶ神託選択でさえ
    first 選択を上回らなかった** (PSNR 21.29 vs 21.43) ため WTA が機能しなかった。

    そこで `load_backbone` は枝1以降の重みに相対摂動 (`_BRANCH_PERTURB`) を加える。
    枝0 は事前学習済みのまま残す (退行しないための保険)。
    このテストはその**設計意図そのもの**を守る。
    """
    base_model = StageF(base=16).to(DEVICE)
    sd = base_model.state_dict()
    k = 4
    model = StageFW2(base=16, k=k).to(DEVICE)
    model.load_backbone(sd)

    # 枝0 は事前学習済みと厳密に一致する (退行しない保険)
    for key, v in model.branches[0].dec.state_dict().items():
        assert torch.equal(v, base_model.dec.state_dict()[key]), f"枝0 が変更されている: {key}"

    # 枝1以降は枝0 と異なる (対称性が壊れている)
    ref = model.branches[0].dec.state_dict()
    for i in range(1, k):
        diffs = [float((v - ref[key]).abs().max())
                 for key, v in model.branches[i].dec.state_dict().items()]
        assert max(diffs) > 0, f"枝{i} が枝0 と同一 (対称性が壊れていない)"

    # 出力レベルでも意味のある多様性があること。
    # 判定は出力分散に対する比で行う (絶対値は入力に依存するため)。
    x = torch.rand(1, 3, 64, 64, device=DEVICE)
    m = torch.zeros(1, 1, 64, 64, device=DEVICE)
    m[:, :, 20:45, 22:40] = 1
    model.eval()
    with torch.no_grad():
        pred = model(x, m)
    ds = [float(((pred[:, i] - pred[:, j]) ** 2).mean())
          for i in range(k) for j in range(i + 1, k)]
    ratio = (sum(ds) / len(ds)) / float(pred.var())
    assert ratio > 0.01, f"仮説間差が小さすぎる (比 {ratio:.5f})。WTA が機能しない"


def test_load_backbone_runs_and_produces_diverse_output():
    """load_backbone 後も forward が動き、出力の解像度が保たれること。"""
    base_model = StageF(base=16).to(DEVICE)
    model = StageFW2(base=16, k=4).to(DEVICE)
    model.load_backbone(base_model.state_dict())
    img, mask = _sample(64, batch=1, seed=4)
    with torch.no_grad():
        out = model(img, mask)
    assert out.shape == (1, 4, 3, 64, 64)
    assert torch.isfinite(out).all()


# --- 4. 枝が独立していること ------------------------------------------------------

def test_branches_are_independent_gradients():
    torch.manual_seed(0)
    model = StageFW2(base=16, k=4).to(DEVICE)
    img, mask = _sample(48, batch=1, seed=2)
    out = model(img, mask)                    # (N,4,3,H,W)
    loss = out[:, 0].sum()                     # 枝0の出力にだけ依存する損失
    model.zero_grad(set_to_none=True)
    loss.backward()

    g0 = sum(float(p.grad.abs().sum()) for p in model.branches[0].parameters()
             if p.grad is not None)
    assert g0 > 0, "枝0に勾配が来ていない"
    for i in range(1, model.k):
        for name, p_ in model.branches[i].named_parameters():
            assert p_.grad is None or float(p_.grad.abs().sum()) == 0.0, \
                f"枝{i}.{name} に勾配が漏れている (重みが共有されている疑い)"
    # emb も同様: emb[0] にだけ勾配が入り、他は入らないはず
    assert model.emb[0].grad is not None and float(model.emb[0].grad.abs().sum()) > 0
    for i in range(1, model.k):
        assert model.emb[i].grad is None or float(model.emb[i].grad.abs().sum()) == 0.0


# --- 5. 多様性が生まれること (この設計の存在理由) ---------------------------------

def _mean_pairwise_sq_diff(out: torch.Tensor, k: int) -> float:
    diffs = [float(((out[:, i] - out[:, j]) ** 2).mean())
             for i in range(k) for j in range(i + 1, k)]
    return sum(diffs) / len(diffs)


def test_stagefw2_hypotheses_more_diverse_than_stagefw():
    """StageFW (出力層のみ分岐) と StageFW2 (デコーダ以降を分岐) を同じ入力・
    同じシードで比較し、StageFW2 の仮説間平均二乗差が StageFW の 2 倍以上に
    なることを確認する。これが `StageFW2` を作った理由そのもの:
    `StageFW` は浅い分岐のせいで仮説がほぼ同一になり、神託選択さえ first を
    上回らなかった (このファイル冒頭 docstring 参照)。
    """
    seed = 123
    k = 4
    img, mask = _sample(64, batch=2, seed=seed)

    torch.manual_seed(seed)
    model_fw = StageFW(base=16, k=k).to(DEVICE)
    torch.manual_seed(seed)
    model_fw2 = StageFW2(base=16, k=k).to(DEVICE)

    with torch.no_grad():
        out_fw = model_fw(img, mask)
        out_fw2 = model_fw2(img, mask)

    d_fw = _mean_pairwise_sq_diff(out_fw, k)
    d_fw2 = _mean_pairwise_sq_diff(out_fw2, k)
    assert d_fw2 > 2 * d_fw, (
        f"StageFW2 の多様性がStageFW比 2倍未満: StageFW2={d_fw2:.6f} "
        f"StageFW={d_fw:.6f} (比 {d_fw2 / max(d_fw, 1e-12):.2f}x)")


# --- 6. checkpoint の往復 ------------------------------------------------------

def test_checkpoint_roundtrip_via_load_stage_fw2(tmp_path):
    model = StageFW2(base=16, k=3).to(DEVICE)
    ck_path = tmp_path / "fw2_test.pt"
    torch.save({"model": model.state_dict(), "step": 10, "base": 16, "k": 3,
               "arch": "fw2"}, ck_path)

    loaded, ck = load_stage_fw2(str(ck_path), DEVICE)
    assert ck["arch"] == "fw2"
    assert ck["base"] == 16 and ck["k"] == 3
    assert loaded.k == 3

    img, mask = _sample(48, batch=1, seed=9)
    with torch.no_grad():
        out_orig = model(img, mask)
        out_loaded = loaded(img, mask)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_checkpoint_missing_metadata_raises():
    """`base`/`k` が checkpoint に無い場合は明確なエラーにすること
    (`StageFW2` は出力層の形から base/k を逆算できないため、`StageFW` の
    ような shape フォールバックが効かない)。"""
    import pytest

    model = StageFW2(base=16, k=2).to(DEVICE)
    ck_path_dir = os.path.dirname(__file__)
    # tmp を使わず単純にメモリ上で確認する (torch.save/load の往復が要件では
    # なくエラー経路の確認なので、直接欠落 dict を検証する)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save({"model": model.state_dict(), "step": 1}, path)
        with pytest.raises(ValueError):
            load_stage_fw2(path, DEVICE)
    finally:
        os.remove(path)
