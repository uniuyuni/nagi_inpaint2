"""`eval/quickval.py` と `eval/benchmark.py` の一致テスト。device は "cpu" 固定。

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

原因は構造的で、**学習時の検証と最終評価が別実装・別データ分布**だったこと。
`eval/quickval.py` の `quick_eval` は `eval/benchmark.py` と全く同じ合成経路
(`sbs.inpaint` -> `stage_d.harmonize` -> `np.where`) と全く同じ指標コード
(`eval/metrics.py` の `report`) を呼ぶことでこれを潰す設計になっている。

**このテストの役割**: 「本当に同じコードを通っているか」を実測で担保する。
`quick_eval` の返す psnr と、同じ2ケースを `eval/benchmark.py` の `run()` へ
渡した結果の psnr が一致しなければ、`quick_eval` は看板どおりに動いていない
ということであり、また今回の作業自体が同じ失敗の5回目になってしまう。
**これが「乖離しない」ことの担保なので必ず入れること** (タスク仕様より)。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import benchmark as bench       # noqa: E402
from eval.quickval import quick_eval      # noqa: E402
from nagi_inpaint2.stage_f import StageF  # noqa: E402

DEVICE = "cpu"
SPEC_PATH = "cache/bench_person_val.json"
CKPT_PATH = "checkpoints/stage_f96/last.pt"
N_CASES = 2

# quick_eval の既定値 (eval/quickval.py の quick_eval シグネチャの既定と同じ)。
# 両方の呼び出しでこの値を明示的に揃えることで、パラメータの取り違えによる
# 差異と「実装そのものがずれている」差異を切り分ける。
TOPK = 8
ADAPTIVE = True
D0 = 4.0
D1 = 16.0

pytestmark = pytest.mark.skipif(
    not (os.path.exists(SPEC_PATH) and os.path.exists(CKPT_PATH)),
    reason=f"{SPEC_PATH} または {CKPT_PATH} が無い (このリポジトリの標準配置が前提)")


@pytest.fixture(scope="module")
def tmp_spec(tmp_path_factory):
    """`cache/bench_person_val.json` の先頭2件だけを書き出した一時 spec。

    テスト後に消す (タスク仕様より)。`tmp_path_factory` (pytest 標準の一時
    ディレクトリ) を使うので、テストプロセス終了時にも残らない。
    """
    spec = json.load(open(SPEC_PATH))
    spec2 = dict(spec)
    spec2["cases"] = spec["cases"][:N_CASES]
    d = tmp_path_factory.mktemp("quickval")
    path = os.path.join(d, "spec2.json")
    with open(path, "w") as f:
        json.dump(spec2, f)
    yield str(path)
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture(scope="module")
def stage_f():
    ck = torch.load(CKPT_PATH, map_location=DEVICE)
    base = int(ck.get("base") or ck["model"]["out.weight"].shape[1])
    sf = StageF(base=base).to(DEVICE)
    sf.load_state_dict(ck["model"])
    sf.eval()
    return sf


def test_quickval_matches_benchmark_psnr(tmp_spec, stage_f):
    """同じモデル・同じ2ケースで、quick_eval と benchmark.run() の psnr が一致する。

    `eval/benchmark.py` の `_build_query` の `src="stagef"` 経路
    (`torch.where(mk > 0.5, sf(x, mk), x)`) と同じクエリを `quick_eval` に渡す。
    重いので `--no-lama` 相当 (LaMa を読み込まない、`with_lama=False`) で行う。
    """
    def query_fn(x, mk):
        return torch.where(mk > 0.5, stage_f(x, mk), x)

    qv = quick_eval(query_fn, spec_path=tmp_spec, n_cases=N_CASES, device=DEVICE,
                    topk=TOPK, adaptive=ADAPTIVE, d0=D0, d1=D1,
                    shard_dir="shards")

    acc = bench.run("shards", tmp_spec, stage_a=None, with_lama=False,
                    device=DEVICE, query_src="stagef", topk=TOPK,
                    stage_f=CKPT_PATH, adaptive=ADAPTIVE, d0=D0, d1=D1)

    # bench.run() は acc = {k: [] for k in names} を names の順で作るので、
    # 挿入順は必ず ["我々(クエリなし)", tag, tag+"+StageD", standalone,
    # standalone+"+StageD"] (with_lama=False, query_src="stagef" の場合)。
    # tag+"+StageD" (クエリ+StageB+StageD 経路、quick_eval と同一の合成経路)
    # がインデックス2。
    keys = list(acc.keys())
    target_key = keys[2]
    assert target_key.endswith("+StageD")
    assert "StageF" in target_key and "クエリなし" not in target_key

    bench_psnr = float(np.median([r["psnr"] for r in acc[target_key]
                                  if np.isfinite(r.get("psnr", np.nan))]))

    print(f"\n[quickval vs benchmark] quickval psnr={qv['psnr']:.6f}  "
          f"benchmark psnr={bench_psnr:.6f}  (key={target_key!r})")
    assert np.isfinite(qv["psnr"])
    assert qv["psnr"] == pytest.approx(bench_psnr, abs=1e-4)
