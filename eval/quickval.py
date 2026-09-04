"""学習中の検証を、最終評価 (`eval/benchmark.py`) と全く同じコードで行う。

**なぜこれが要る (背景)**

このセッションで**同じ種類の失敗を4回**繰り返した。いずれも「学習中に見ていた
数値が、最終評価の数値と乖離していた」もの:

  - BatchNorm の running_var を「パラメータが動いている」と誤読した
  - 照合コスト学習で cache-val の regret は 1.481→1.236 と改善したのに、
    下流の PSNR は -2.1 dB 悪化した
  - **直近**: WTA 学習の検証が `grain 1.021` (理想 1.0) を報告したが、
    人物マスクで測ると **0.098** だった。学習側の検証が**混合マスク (電線など
    細いもの)** で測っていたため。細い穴では grain 比が自動的に 1.0 付近になり、
    **ぼけを検出できない** (`tests/test_metrics_sanity.py` の
    `test_grain_is_blind_on_thin_masks` に再現テストを残してある)。

原因は構造的で、**学習時の検証と最終評価が別実装・別データ分布**だったこと。
本モジュールは `eval/benchmark.py` と全く同じ合成経路
(`sbs.inpaint` -> `stage_d.harmonize` -> `np.where`) と全く同じ指標コード
(`eval/metrics.py` の `report`) を学習ループから直接呼べるようにすることで、
この構造的乖離を潰す。二重実装をやめる、という一点がすべて。

`quick_eval()` は既定で `cache/bench_person_val.json` (人物サイズの
silhouette マスクで構成された保留ケース集合) の**先頭 n_cases 件を固定**で
使う。乱数は一切使わない (毎回同じケースで比較できるようにするため。
学習経過を追うグラフが乱数のノイズで揺れると、改善したのか誤差なのか
判断できなくなる)。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.benchmark import load_shard_tiles   # noqa: E402
from eval.metrics import report               # noqa: E402
from masks import sample_mask                 # noqa: E402
from nagi_inpaint2 import stage_b_search as sbs  # noqa: E402
from nagi_inpaint2 import stage_d                # noqa: E402

# shard タイルの読み込みキャッシュ (モジュールレベル)。
#
# `quick_eval` は学習ループから `--val-every` 毎に繰り返し呼ばれる。
# `load_shard_tiles` は tar を開いて zstd 展開する重い処理なので、毎回
# 読み直すと学習が大きく遅くなる。同じ shard パスは一度読めば内容は
# 変わらない (保留 shard は固定ファイル) ので、プロセス内で使い回す。
_TILE_CACHE: dict[str, dict] = {}


def _tiles_for_shard(shard_path: str) -> dict:
    if shard_path not in _TILE_CACHE:
        _TILE_CACHE[shard_path] = {
            k: (img, satm, meta)
            for k, img, satm, meta in load_shard_tiles(shard_path, stride=1)
        }
    return _TILE_CACHE[shard_path]


def quick_eval(query_fn, spec_path: str = "cache/bench_person_val.json",
               n_cases: int = 10, device: str = "mps", topk: int = 8,
               adaptive: bool = True, d0: float = 4.0, d1: float = 16.0,
               perceptual=None, shard_dir: str = "shards") -> dict:
    """`eval/benchmark.py` と同一の合成経路・指標コードで学習中のモデルを検証する。

    `query_fn(x, mk) -> (1,3,H,W)`: 学習中のモデルからクエリ (穴を埋めた画像)
    を作る呼び出し可能オブジェクト。`eval/benchmark.py` の `_build_query` の
    `src="stagef"` 経路 (`torch.where(mk > 0.5, sf(x, mk), x)`) に相当するものを
    呼び出し側 (学習スクリプト) が組み立てて渡す。

    `spec_path` の**先頭 `n_cases` 件**を固定で使う (毎回同じケース。乱数は
    使わない)。合成経路は `eval/benchmark.py` の `run()` の
    "クエリ+StageB+StageD" 経路と同一:
    `sbs.inpaint(x, mk, query=q, topk=..., adaptive=..., d0=..., d1=...)`
    -> `stage_d.harmonize(..., do_color=False, do_poisson=True, do_grain=False)`
    -> `np.where(mask, out, img)`。

    指標は `eval/metrics.py` の `report()` をそのまま使う (自前で書き直さない)。
    `perceptual` (`eval.perceptual.Perceptual` インスタンス) が渡されていれば
    `lpips` も足す。

    戻り値: 各指標の**中央値** dict。`{"boundary":…, "grain":…, "psd":…,
    "psnr":…}` (`perceptual` を渡した場合はさらに `"lpips"` も含む)。
    """
    spec = json.load(open(spec_path))
    cases = spec["cases"][:n_cases]

    rows: list[dict] = []
    for c in cases:
        shard_path = os.path.join(shard_dir, c["shard"])
        tiles = _tiles_for_shard(shard_path)
        if c["key"] not in tiles:
            continue
        img, satm, _ = tiles[c["key"]]
        mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                              kind=c["kind"])
        x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(device)
        mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)

        with torch.inference_mode():
            q = query_fn(x, mk)
            o1 = sbs.inpaint(x, mk, query=q, topk=topk, adaptive=adaptive,
                             d0=d0, d1=d1)[0].permute(1, 2, 0).cpu().numpy()
        d = stage_d.harmonize(o1, img, mask, valid=~satm,
                              rng=np.random.default_rng(0),
                              do_color=False, do_poisson=True, do_grain=False)
        full = np.where(mask[..., None], d, img)

        r = report(full, img, mask, gt=img)
        if perceptual is not None:
            r["lpips"] = perceptual(full, img, mask=mask)
        rows.append(r)

    def med(k: str) -> float:
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    keys = ["boundary", "grain", "psd", "psnr"]
    if perceptual is not None:
        keys.append("lpips")
    return {k: med(k) for k in keys}
