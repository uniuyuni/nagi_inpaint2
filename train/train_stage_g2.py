"""Stage G2 の学習 — 残差の局所スケール正規化版 (`train/train_stage_g.py` の後継)。

`train/train_stage_g.py` を出発点にする。**目的はモデルの完成ではなく問いに
答えること** (前回の学習と同じ理由: 「効く理屈」が実際に効くか)。

**効く理屈と、平坦部が悪化した実例の引用は `nagi_inpaint2/stage_g.py` の
`residual_scale` 周辺の docstring を参照**。ここでは学習ループに固有の要点
だけを書く:

  - `StageF` は前回と同じく**凍結**して使う。Stage G2 が学習するのは
    正規化残差 `r_n = r / (scale + eps)` だけ。
  - 条件は 9ch (`stage_g.make_cond9`)。前回の 8ch (`make_cond`) に
    `residual_scale` の出力 `scale` を1ch追加しただけで、他のチャンネルの
    意味は変えない (`copy_resid` の位置付けなども前回と同じ)。
  - 損失は `stage_g.rectified_flow_loss_scaled`、サンプリングは
    `stage_g.sample_residual_scaled` を使う (いずれも既存の
    `rectified_flow_loss` / `sample_residual` に正規化の前処理・後処理を
    足しただけで、中身の積分ロジックは共有している — 二重実装をしない)。
  - **前回と同じ step 数 (既定 3000) にする**。設計変更 (正規化の有無) の
    効果だけを比較したいので、学習量という別の変数を混ぜない。

**検証は人物マスクで固定ケースを使う (学習ローダーの混合マスクではない)**。
`cache/bench_person_val.json` の先頭 8 ケースを毎回同じ画像・同じマスクで
評価する (`eval/benchmark.py` の `load_shard_tiles` / `data/masks.py` の
`sample_mask` を使い、マスクの種は `mask_seed` で固定するので毎回同じマスク
形状になる)。前回 (`train_stage_g.py`) の grain ログに加えて、**平坦部だけを
切り出した振幅の比較**を出す:

    [平坦部] StageF単体の残差振幅=X.XXXXX  StageGの追加振幅=Y.YYYYX

平坦部の定義は「そのケースのマスク周辺 (annulus) の高周波 RMS が、
8ケース中央値未満」。この値が小さいケース群で、StageG が追加する振幅
(`sample_residual_scaled` の出力の穴内 std) が小さいままであることが、
今回の設計変更 (局所スケール正規化) の成否を直接示す — 大きいままなら
前回と同じ「平坦部の斑模様」がまだ起きていることになる。

過学習ゲート (`--overfit`) は前回と同じ理由で必須:

  .pixi/envs/default/bin/python train/train_stage_g2.py --overfit 2 --steps 60 \\
      --crop 128 --device cpu --log-every 10 --val-every 60 --use-copy-cond 0 \\
      --out /tmp/g2_gate
  .pixi/envs/default/bin/python train/train_stage_g2.py --steps 3000

**GPU (mps) は使わないこと。本学習はこのタスクでは実行しない** (実装と CPU
過学習ゲートの確認までがスコープ)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.benchmark import load_shard_tiles                       # noqa: E402
from masks import sample_mask                                      # noqa: E402
from nagi_inpaint2 import stage_b_search as sbs                   # noqa: E402
from nagi_inpaint2.stage_f import StageF                          # noqa: E402
from nagi_inpaint2.stage_g import (StageG, make_cond9,             # noqa: E402
                                   rectified_flow_loss_scaled,
                                   residual_scale, sample_residual_scaled)
from train.dataset import list_shards, make_loader, split_shards  # noqa: E402
from train.train_stage_f import random_crop                       # noqa: E402
from train.train_stage_fw import _dilate4_torch, _highpass_torch  # noqa: E402
from train.train_stage_fw import grain_ratio_torch                # noqa: E402
from train.train_stage_g import load_frozen_stage_f                # noqa: E402


def p(*a):
    print(*a, flush=True)


@torch.no_grad()
def compute_r_cond_scaled(sf: StageF, img: torch.Tensor, mask: torch.Tensor,
                          use_copy_cond: bool):
    """`img`/`mask` から学習対象 `r`、条件 `cond` (9ch)、`scale` を組み立てる。

    `train_stage_g.compute_r_cond` と同じ手順に、`residual_scale` の呼び出しと
    `make_cond9` への差し替えを足しただけ (StageF・Stage B 検索の使い方は
    一切変えない)。

    **漏洩について**: `residual_scale(img, mask)` にはここでも `img` (穴の中に
    正解が入ったままの画像) をそのまま渡す。これは問題ない —
    `residual_scale` 自身が既知画素だけを使うよう設計されており
    (`nagi_inpaint2/stage_g.py` の docstring と `tests/test_stage_g2.py` の
    漏洩テストを参照)、呼び出し側で穴を隠す必要はない。
    """
    sf_out = sf(img, mask)
    r = img - sf_out
    dist = sbs.hole_distance(mask)
    if use_copy_cond:
        idx, offs, _ = sbs.solve_blocks(img, mask, query=sf_out)
        copy = sbs.paste(img, mask, idx, offs)
        copy_resid = torch.where(mask > 0.5, copy - sf_out, torch.zeros_like(sf_out))
    else:
        copy_resid = torch.zeros_like(sf_out)
    scale = residual_scale(img, mask)
    cond = make_cond9(sf_out, mask, dist, copy_resid, scale)
    return r, cond, sf_out, scale


# --- 保留 shard のタイルをモジュールレベルでキャッシュする -------------------
#
# `cache/bench_person_val.json` の固定8ケースは学習中に何度も (val-every ごと
# に) 評価する。`load_shard_tiles` は tar から zstd 展開する重い処理なので、
# shard ごとに1回だけ実行してキャッシュする (毎回展開すると学習が不要に遅くなる)。
_SHARD_TILE_CACHE: dict[str, dict] = {}


def _get_shard_tiles(shard_path: str) -> dict:
    if shard_path not in _SHARD_TILE_CACHE:
        _SHARD_TILE_CACHE[shard_path] = {
            k: (im, sm) for k, im, sm, _ in load_shard_tiles(shard_path, stride=1)}
    return _SHARD_TILE_CACHE[shard_path]


def _load_person_cases(shards_dir: str, spec_path: str, n_cases: int):
    """`cache/bench_person_val.json` の先頭 `n_cases` 件を (case, img, mask) に読む。

    マスクは `mask_seed` から再構成するので、呼び出すたびに同じ形状になる
    (`eval/stage_g_visual.py` と同じ手続き)。
    """
    spec = json.load(open(spec_path))
    cases = spec["cases"][:n_cases]
    out = []
    for c in cases:
        tiles = _get_shard_tiles(os.path.join(shards_dir, c["shard"]))
        if c["key"] not in tiles:
            continue
        img, _satm = tiles[c["key"]]
        mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                              kind=c["kind"])
        out.append((c, img, mask))
    return out


@torch.no_grad()
def validate_person(model: StageG, sf: StageF, shards_dir: str, spec_path: str,
                    device: str, use_copy_cond: bool, sample_steps: int,
                    n_cases: int = 8, annulus: int = 32):
    """`cache/bench_person_val.json` の固定8ケースで grain 比と平坦部振幅を測る。

    **人物マスクで固定して測る** — 学習ローダーの混合マスク (wire/dust/rect
    なども混じる) では grain が nan になりやすく判定に使えない
    (`nagi_inpaint2/stage_g.py` の docstring 参照)。

    平坦部の定義は「そのケースのマスク周辺 annulus の高周波 RMS が、8ケース
    中央値未満」。`grain_ratio_torch` 内部の annulus 計算 (`_dilate4_torch` /
    `_highpass_torch`) と全く同じ核を使って、判定基準がそこと食い違わない
    ようにする。

    戻り値: dict(grain_f, grain_g, flat_resid_f, flat_resid_g, n_flat)
      - grain_f / grain_g : `StageF` 単体 / `+StageG` の grain 比の平均 (nan は除く)
      - flat_resid_f       : 平坦ケースにおける「StageF単体の残差振幅」
                              (= (img - sf_out) の穴内 std の平均)
      - flat_resid_g       : 平坦ケースにおける「StageGの追加振幅」
                              (= sample_residual_scaled の穴内 std の平均)
      - n_flat              : 平坦判定されたケース数
    """
    model.eval()
    cases = _load_person_cases(shards_dir, spec_path, n_cases)

    hf_list = []
    per_case = []
    for i, (c, img, mask) in enumerate(cases):
        x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(device)
        mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
        if mk.sum() < 1:
            continue

        r, cond, sf_out, scale = compute_r_cond_scaled(sf, x, mk, use_copy_cond)
        gen = torch.Generator().manual_seed(1000 + i)
        r_hat = sample_residual_scaled(model, cond, mk, scale, steps=sample_steps,
                                       generator=gen)
        out = sf_out + r_hat

        gf = grain_ratio_torch(torch.where(mk > 0.5, sf_out, x), mk, annulus=annulus)
        gg = grain_ratio_torch(torch.where(mk > 0.5, out, x), mk, annulus=annulus)

        # マスク周辺 annulus の高周波 RMS (`grain_ratio_torch` と同じ核を使う)
        g = x.mean(dim=1, keepdim=True)
        hp = _highpass_torch(g)
        m = (mk > 0.5).float()
        ann_b = (_dilate4_torch(m, annulus) * (1.0 - m)) > 0.5
        hole_b = m > 0.5
        hf_ann = float(hp[ann_b].std()) if int(ann_b.sum()) >= 64 else float("nan")

        resid_f = float((x - sf_out)[hole_b.expand_as(x)].std()) if int(hole_b.sum()) > 0 \
            else float("nan")
        resid_g = float(r_hat[hole_b.expand_as(r_hat)].std()) if int(hole_b.sum()) > 0 \
            else float("nan")

        hf_list.append(hf_ann)
        per_case.append(dict(gf=gf, gg=gg, hf_ann=hf_ann, resid_f=resid_f, resid_g=resid_g))

    model.train()

    finite_hf = [h for h in hf_list if np.isfinite(h)]
    med_hf = float(np.median(finite_hf)) if finite_hf else float("nan")

    def _mean(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    grain_f = _mean([c["gf"] for c in per_case])
    grain_g = _mean([c["gg"] for c in per_case])

    flat_cases = [c for c in per_case if np.isfinite(c["hf_ann"]) and c["hf_ann"] < med_hf]
    flat_resid_f = _mean([c["resid_f"] for c in flat_cases])
    flat_resid_g = _mean([c["resid_g"] for c in flat_cases])

    return dict(grain_f=grain_f, grain_g=grain_g, flat_resid_f=flat_resid_f,
               flat_resid_g=flat_resid_g, n_flat=len(flat_cases), n_cases=len(per_case))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256,
                    help="全解像度 (512) がメモリに乗らない場合のランダムクロップ辺")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32, help="StageG の base")
    ap.add_argument("--stage-f", default="checkpoints/stage_f96/last.pt",
                    help="凍結して使う StageF の checkpoint")
    ap.add_argument("--use-copy-cond", type=int, default=1,
                    help="1: Stage B の検索パッチ残差を条件に含める (重い)。"
                         "0: 条件チャンネルを0で埋める (速度比較用)")
    ap.add_argument("--sample-steps", type=int, default=8,
                    help="検証時の sample_residual_scaled のオイラー法ステップ数")
    ap.add_argument("--mask-kind", default=None)
    ap.add_argument("--person-spec", default="cache/bench_person_val.json",
                    help="固定検証ケース集合 (人物マスク)")
    ap.add_argument("--person-cases", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/stage_g2")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    cond_ch = 9   # 固定 (仕様どおり)。StageG2 は必ず 9ch 条件で学習する。

    tr, va = split_shards(list_shards(a.shards))       # 無作為分割 (学習ローダー用)
    del va
    dl = make_loader(tr, batch_size=a.batch, workers=a.workers,
                     mask_kind=a.mask_kind)
    it = iter(dl)
    p(f"shard: 学習 {len(tr)} 本")

    sf = load_frozen_stage_f(a.stage_f, a.device)

    model = StageG(base=a.base, cond_ch=cond_ch).to(a.device)
    n_par = sum(q.numel() for q in model.parameters())
    p(f"StageG params: {n_par:,}  (base={a.base}, cond_ch={cond_ch}, crop={a.crop}, "
      f"use_copy_cond={bool(a.use_copy_cond)})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    w0 = {k: v.detach().clone() for k, v in model.named_parameters()}

    fixed = None
    if a.overfit:
        # **穴が空 (mask.sum()==0) の抽選を引かないよう引き直す。**
        # (`train_stage_g.py` と同じ理由・同じ事故の再発防止)
        for _ in range(50):
            b = next(it)
            img, mask = b["image"][:a.overfit].to(a.device), b["mask"][:a.overfit].to(a.device)
            img, mask = random_crop(img, mask, a.crop)
            if mask.sum() >= 1:
                break
        else:
            raise RuntimeError("50 回引き直しても穴のあるタイルが取れなかった")
        fixed = compute_r_cond_scaled(sf, img, mask, a.use_copy_cond)
        p(f"過学習テスト: 固定 {a.overfit} タイル (crop={a.crop}, "
          f"mask画素数={int(mask.sum())})")

    hist = []
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        if fixed is not None:
            r, cond, _sf_out, scale = fixed
        else:
            b = next(it)
            img, mask = b["image"].to(a.device), b["mask"].to(a.device)
            img, mask = random_crop(img, mask, a.crop)
            if mask.sum() < 1:
                continue
            r, cond, _sf_out, scale = compute_r_cond_scaled(sf, img, mask, a.use_copy_cond)

        loss, parts = rectified_flow_loss_scaled(model, r, cond, scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % a.log_every == 0:
            with torch.no_grad():
                dw = max(float((q - w0[k]).abs().max())
                         for k, q in model.named_parameters())
            el = time.perf_counter() - t0
            rec = dict(step=step, loss=float(loss.detach()), dw=dw,
                       lr=float(sched.get_last_lr()[0]), sec_per_step=el / step,
                       **parts)
            hist.append(rec)
            p(f"  {step:>6}/{a.steps}  loss {rec['loss']:.5f}  |Δw|max {dw:.3e}  "
              f"{rec['sec_per_step']*1000:.0f}ms/step")

        if step % a.val_every == 0 or step == a.steps:
            v = validate_person(model, sf, a.shards, a.person_spec, a.device,
                               bool(a.use_copy_cond), a.sample_steps, a.person_cases)
            if hist:
                hist[-1]["val_grain_f"] = v["grain_f"]
                hist[-1]["val_grain_g"] = v["grain_g"]
                hist[-1]["val_flat_resid_f"] = v["flat_resid_f"]
                hist[-1]["val_flat_resid_g"] = v["flat_resid_g"]
            p(f"  [grain] StageF単体={v['grain_f']:.3f}  +StageG={v['grain_g']:.3f}  "
              f"(理想 1.0)")
            p(f"  [平坦部] StageF単体の残差振幅={v['flat_resid_f']:.5f}  "
              f"StageGの追加振幅={v['flat_resid_g']:.5f}  "
              f"(平坦ケース {v['n_flat']}/{v['n_cases']})")

        if step % 1000 == 0 or step == a.steps:
            os.makedirs(a.out, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step, "base": a.base,
                       "cond_ch": cond_ch},
                       os.path.join(a.out, "last.pt"))
            with open(os.path.join(a.out, "history.jsonl"), "w") as f:
                for r_ in hist:
                    f.write(json.dumps(r_) + "\n")

    if len(hist) >= 2:
        p("\n=== 学習前後 ===")
        p(f"  loss     {hist[0]['loss']:.5f} -> {hist[-1]['loss']:.5f} "
          f"({'改善' if hist[-1]['loss'] < hist[0]['loss'] * 0.9 else '**改善が弱い**'})")
        p(f"  |Δw|max  {hist[-1]['dw']:.3e} "
          f"({'パラメータが動いている' if hist[-1]['dw'] > 1e-5 else '**動いていない**'})")
        gs = [(r_["step"], r_.get("val_grain_f"), r_.get("val_grain_g"))
              for r_ in hist if r_.get("val_grain_g") is not None]
        if gs:
            p(f"  grain (StageF単体 -> +StageG)  {gs[0][1]:.3f} -> {gs[-1][1]:.3f}  /  "
              f"{gs[0][2]:.3f} -> {gs[-1][2]:.3f} (理想 1.0)")
        fr = [(r_["step"], r_.get("val_flat_resid_f"), r_.get("val_flat_resid_g"))
              for r_ in hist if r_.get("val_flat_resid_g") is not None]
        if fr:
            p(f"  平坦部振幅 (StageF単体 -> StageG追加分)  "
              f"{fr[0][1]:.5f} -> {fr[-1][1]:.5f}  /  {fr[0][2]:.5f} -> {fr[-1][2]:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
