"""Stage G の学習 — 残差 (StageF の取りこぼし) を Rectified Flow で標本として引く。

`train/train_stage_f.py` を出発点にする。**目的はモデルの完成ではなく問いに
答えること** (G1: 実現可能性検証) — この点は StageF/StageFW の学習スクリプトと同じ。

**効く理屈と実測の引用は `nagi_inpaint2/stage_g.py` の docstring を参照。** ここでは
学習ループに固有の要点だけを書く:

  - `StageF` は**凍結**して使う (`eval()` + `requires_grad_(False)`)。Stage G が学習
    するのは残差だけであり、StageF の出力自体は変えない (実現可能性検証なので
    変数を1つに絞る)。
  - `r = img - sf_out` が学習対象。`copy_resid = copy - sf_out` は Stage B の検索
    パッチとの差分で、条件チャンネルの1つに過ぎない (`stage_g.make_cond` 参照。
    残差との相関 +0.210 は「決定的ではない」ので、モデルが無視できる補助入力と
    して与えるだけで、特別な合成をしない)。
  - 検索 (`sbs.solve_blocks`) は重いので `--use-copy-cond 0` で外せる。外した場合は
    そのチャンネルを 0 で埋める (`stage_g.make_cond` の契約どおり)。
  - **grain がこの実験の主目的**。`StageF出力 + サンプリング残差` の grain が
    `StageF` 単体より明確に 1.0 に近づかなければ、この設計は成立しない
    (`nagi_inpaint2/stage_g.py` の「効かない条件」参照)。`grain_ratio_torch` は
    `train/train_stage_fw.py` から import する (二重実装をしない)。

過学学習ゲート (`--overfit`) は StageA/StageF/StageFW と同じ理由で必須:

  .pixi/envs/default/bin/python train/train_stage_g.py --overfit 2 --steps 60 \\
      --crop 128 --device cpu --log-every 10 --val-every 60 --use-copy-cond 0 \\
      --out /tmp/g_gate
  .pixi/envs/default/bin/python train/train_stage_g.py --steps 3000
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

from nagi_inpaint2 import stage_b_search as sbs                   # noqa: E402
from nagi_inpaint2.stage_f import StageF                          # noqa: E402
from nagi_inpaint2.stage_g import (StageG, make_cond,              # noqa: E402
                                   rectified_flow_loss, sample_residual)
from train.dataset import list_shards, make_loader, split_shards  # noqa: E402
from train.train_stage_f import random_crop                       # noqa: E402
from train.train_stage_fw import grain_ratio_torch                # noqa: E402


def p(*a):
    print(*a, flush=True)


def load_frozen_stage_f(path: str, device: str) -> StageF:
    """`checkpoints/.../last.pt` から StageF を読み込み凍結する。

    base は checkpoint の `out.weight` の形から復元する
    (`train_stage_fw.py --init` と同じ流儀)。
    """
    ck = torch.load(path, map_location=device)
    base = int(ck.get("base") or ck["model"]["out.weight"].shape[1])
    sf = StageF(base=base).to(device)
    sf.load_state_dict(ck["model"])
    sf.eval()
    sf.requires_grad_(False)
    p(f"  StageF (凍結): {path} (step {ck.get('step')}, base {base})")
    return sf


@torch.no_grad()
def compute_r_cond(sf: StageF, img: torch.Tensor, mask: torch.Tensor,
                   use_copy_cond: bool):
    """`img`/`mask` から学習対象 `r` と条件 `cond` を組み立てる。

    StageF・Stage B の検索はいずれも勾配を必要としない (前者は凍結、後者は
    非パラメトリック) ので `@torch.no_grad()` で丸ごと計算する。

    `copy_resid` を作るときの `query` には `sf_out` を使う — `solve_blocks` は
    穴の内部を照合するために穴を埋めた画像を要求する
    (`stage_b_search.solve_blocks` の docstring 参照)。StageF の出力がここでは
    最良の埋め (Stage A 相当) になる。
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
    cond = make_cond(sf_out, mask, dist, copy_resid)
    return r, cond, sf_out


@torch.no_grad()
def validate(model: StageG, sf: StageF, it, n: int, device: str, crop: int,
            use_copy_cond: bool, sample_steps: int):
    """保留 shard で grain 比を測る (`train_stage_fw.validate` と同じ考え方)。

    **grain がこの実験の主目的**。`StageF` 単体 (`sf_out`) の grain と
    `sf_out + サンプリング残差` の grain の両方を出す — 比較なしに数値の意味は
    分からない (`quickval.py` docstring の教訓と同じ)。
    """
    model.eval()
    tot_f = tot_g = 0.0
    n_f = n_g = 0
    for _ in range(n):
        b = next(it)
        img, mask = b["image"].to(device), b["mask"].to(device)
        img, mask = random_crop(img, mask, crop)
        if mask.sum() < 1:
            continue
        r, cond, sf_out = compute_r_cond(sf, img, mask, use_copy_cond)
        r_hat = sample_residual(model, cond, mask, steps=sample_steps)
        out = sf_out + r_hat
        for i in range(img.shape[0]):
            mi = mask[i:i + 1]
            gf = grain_ratio_torch(torch.where(mi > 0.5, sf_out[i:i + 1], img[i:i + 1]), mi)
            gg = grain_ratio_torch(torch.where(mi > 0.5, out[i:i + 1], img[i:i + 1]), mi)
            if np.isfinite(gf):
                tot_f += gf
                n_f += 1
            if np.isfinite(gg):
                tot_g += gg
                n_g += 1
    model.train()
    vf = tot_f / n_f if n_f else float("nan")
    vg = tot_g / n_g if n_g else float("nan")
    return vf, vg


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
                    help="検証時の sample_residual のオイラー法ステップ数")
    # マスク種別を固定しない (StageA/StageF と同じ理由: blob だけでは分布が狭く汎化しない)
    ap.add_argument("--mask-kind", default=None)
    ap.add_argument("--out", default="checkpoints/stage_g")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=12)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    tr, va = split_shards(list_shards(a.shards))       # 無作為分割
    dl = make_loader(tr, batch_size=a.batch, workers=a.workers,
                     mask_kind=a.mask_kind)
    it = iter(dl)
    vdl = make_loader(va, batch_size=a.batch, workers=1,
                      mask_kind=a.mask_kind, seed=999)
    vit = iter(vdl)
    p(f"shard: 学習 {len(tr)} / 保留 {len(va)} (無作為)")

    sf = load_frozen_stage_f(a.stage_f, a.device)

    model = StageG(base=a.base, cond_ch=8).to(a.device)
    n_par = sum(q.numel() for q in model.parameters())
    p(f"StageG params: {n_par:,}  (base={a.base}, crop={a.crop}, "
      f"use_copy_cond={bool(a.use_copy_cond)})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    w0 = {k: v.detach().clone() for k, v in model.named_parameters()}

    fixed = None
    if a.overfit:
        # **穴が空 (mask.sum()==0) の抽選を引かないよう引き直す。** crop が
        # マスクを外してしまう抽選もあり得て、そのまま固定すると
        # `rectified_flow_loss` の hole_frac が常に 0 になり、損失が
        # 常に 0/eps ≈ 0 のまま「学習しているように見えて実は何も学習して
        # いない」ことになる (実際に一度この事故を踏んだ)。
        for _ in range(50):
            b = next(it)
            img, mask = b["image"][:a.overfit].to(a.device), b["mask"][:a.overfit].to(a.device)
            img, mask = random_crop(img, mask, a.crop)
            if mask.sum() >= 1:
                break
        else:
            raise RuntimeError("50 回引き直しても穴のあるタイルが取れなかった")
        fixed = compute_r_cond(sf, img, mask, a.use_copy_cond)
        p(f"過学習テスト: 固定 {a.overfit} タイル (crop={a.crop}, "
          f"mask画素数={int(mask.sum())})")

    hist = []
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        if fixed is not None:
            r, cond, _sf_out = fixed
        else:
            b = next(it)
            img, mask = b["image"].to(a.device), b["mask"].to(a.device)
            img, mask = random_crop(img, mask, a.crop)
            if mask.sum() < 1:
                continue
            r, cond, _sf_out = compute_r_cond(sf, img, mask, a.use_copy_cond)

        loss, parts = rectified_flow_loss(model, r, cond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % a.log_every == 0:
            with torch.no_grad():
                # **buffers を含めない。** named_parameters() のみで |Δw|max を計算する
                # (StageA/StageF/StageFW で BatchNorm の running_var 暴走を誤読した教訓)。
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
            vf, vg = validate(model, sf, vit, a.val_batches, a.device, a.crop,
                              a.use_copy_cond, a.sample_steps)
            if hist:
                hist[-1]["val_grain_f"] = vf
                hist[-1]["val_grain_g"] = vg
            p(f"  [grain] StageF単体={vf:.3f}  +StageG={vg:.3f}  (理想 1.0)")

        if step % 1000 == 0 or step == a.steps:
            os.makedirs(a.out, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step, "base": a.base},
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
        gs = [(r_["step"], r_["val_grain_f"], r_["val_grain_g"])
              for r_ in hist if r_.get("val_grain_g") is not None]
        if gs:
            p(f"  grain (StageF単体 -> +StageG)  {gs[0][1]:.3f} -> {gs[-1][1]:.3f}  /  "
              f"{gs[0][2]:.3f} -> {gs[-1][2]:.3f} (理想 1.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
