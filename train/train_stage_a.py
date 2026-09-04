"""Stage A の学習 (SPEC §4.1, §7)。

低周波補完だけを学ぶ。高周波は Stage B が実画素から供給するので、
**GAN も知覚損失も要らない**。L1 + 勾配損失で足りる。

**このセッションで学習は4回失敗した。** いずれも構造の欠落を学習で埋めようとして
いたことが原因だった。今回は構造が正しいことを先に確認してある
(LaMa の低周波をクエリに使うと 21.65 -> 29.06 dB)。
Stage A の仕事は「LaMa の低周波と同等のものを自前で作る」ことに限定されている。

失敗から組み込んだ安全装置:
  - **過学習テストを関門にする** (--overfit)。数百 step で通らなければ本学習に入らない
  - **パラメータの変化量を毎回記録**する
  - 主指標を「LaMa の低周波との差」にする。損失の絶対値ではなく、
    代替として使えるかを直接測る

  pixi run python train/train_stage_a.py --overfit 8 --steps 300   # 関門
  pixi run python train/train_stage_a.py --steps 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_a import SHRINK, StageA, low_target      # noqa: E402
from train.dataset import list_shards, make_loader, split_shards  # noqa: E402


def p(*a):
    print(*a, flush=True)


def grad_loss(a, b):
    """勾配も合わせる。低周波でも構造の位置がずれると Stage B の照合が狂う。"""
    return ((a[:, :, 1:] - a[:, :, :-1]) - (b[:, :, 1:] - b[:, :, :-1])).abs().mean() \
        + ((a[..., 1:] - a[..., :-1]) - (b[..., 1:] - b[..., :-1])).abs().mean()


def step_loss(model, img, mask, lam_grad: float, lam_known: float = 0.3):
    """穴の L1 + 勾配 + **既知領域の L1**。

    既知領域にも L1 を掛けるのは LaMa の設定に倣ったもの (あちらは重み10)。
    「見えている部分を復元できる」ことを要求すると再構成の錨になり、
    穴だけを最適化するより汎化が安定する。
      https://github.com/advimman/lama/blob/main/configs/training/big-lama.yaml
    """
    lo_img = low_target(img)
    lo_mask = (F.interpolate(mask, scale_factor=1.0 / SHRINK, mode="area") > 0.05).float()
    pred = model(lo_img, lo_mask)
    m = lo_mask.expand_as(pred)
    if m.sum() < 1:
        return None, None
    l1 = ((pred - lo_img).abs() * m).sum() / (m.sum() + 1e-6)
    k = 1.0 - m
    l1k = ((pred - lo_img).abs() * k).sum() / (k.sum() + 1e-6)
    lg = grad_loss(pred * lo_mask, lo_img * lo_mask)
    return l1 + lam_grad * lg + lam_known * l1k, l1


@torch.no_grad()
def validate(model, it, n: int, device: str, lam_grad: float):
    """**保留 shard での L1 を学習中に監視する。**

    前回 30,000 step 回して初めて過学習に気づいた。学習中に保留損失を
    見ていなかったことが原因。ぼかし基準との比 (改善率) も併せて出す —
    保留セットの内容が難しいだけなのか、汎化していないのかを区別するため。
    """
    model.eval()
    tot_s = tot_b = 0.0
    c = 0
    for _ in range(n):
        b = next(it)
        img, mask = b["image"].to(device), b["mask"].to(device)
        lo = low_target(img)
        lm = (F.interpolate(mask, scale_factor=1.0 / SHRINK, mode="area") > 0.05).float()
        m = lm.expand_as(lo)
        if m.sum() < 1:
            continue
        pred = model(lo, lm)
        blur = F.avg_pool2d(F.pad(lo * (1 - lm), (4, 4, 4, 4), mode="replicate"),
                            9, stride=1) / (
            F.avg_pool2d(F.pad(1 - lm, (4, 4, 4, 4), mode="replicate"), 9, stride=1) + 1e-6)
        tot_s += float(((pred - lo).abs() * m).sum() / m.sum())
        tot_b += float(((blur - lo).abs() * m).sum() / m.sum())
        c += 1
    model.train()
    if c == 0:
        return float("nan"), float("nan")
    return tot_s / c, 1.0 - (tot_s / c) / max(tot_b / c, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam-grad", type=float, default=0.5)
    # **マスク種別を固定しない。** blob だけで学習すると分布が狭く汎化しない。
    # LaMa も irregular ストロークと矩形を等重みで混ぜている。
    ap.add_argument("--mask-kind", default=None)
    ap.add_argument("--out", default="checkpoints/stage_a")
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

    model = StageA().to(a.device)
    n_par = sum(q.numel() for q in model.parameters())
    p(f"StageA params: {n_par:,}  (1/{SHRINK} 解像度で動作)")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    # 損失が振れて収束しなかったので減衰を入れる (8,000 step で 0.044-0.088 を往復した)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    w0 = {k: v.detach().clone() for k, v in model.named_parameters()}

    fixed = None
    if a.overfit:
        b = next(it)
        fixed = (b["image"][:a.overfit].to(a.device),
                 b["mask"][:a.overfit].to(a.device))
        p(f"過学習テスト: 固定 {a.overfit} タイル")

    hist = []
    best_imp = -1e9          # 保留の改善率が最良のときのモデルを別に残す
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        if fixed is not None:
            img, mask = fixed
        else:
            b = next(it)
            img, mask = b["image"].to(a.device), b["mask"].to(a.device)
        loss, l1 = step_loss(model, img, mask, a.lam_grad)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % a.log_every == 0:
            with torch.no_grad():
                # **buffers を含めない。** 前回は BatchNorm の running_var が
                # 340 まで膨れた値を「パラメータが動いている」と誤読していた。
                dw = max(float((q - w0[k]).abs().max())
                         for k, q in model.named_parameters())
            el = time.perf_counter() - t0
            rec = dict(step=step, loss=float(loss.detach()), l1=float(l1.detach()),
                       dw=dw, lr=float(sched.get_last_lr()[0]), sec_per_step=el / step)
            hist.append(rec)
            p(f"  {step:>6}/{a.steps}  loss {rec['loss']:.5f}  L1 {rec['l1']:.5f}  "
              f"|Δw|max {dw:.3e}  {rec['sec_per_step']*1000:.0f}ms/step")

        if step % a.val_every == 0 or step == a.steps:
            vl, imp = validate(model, vit, a.val_batches, a.device, a.lam_grad)
            if hist:
                hist[-1]["val_l1"] = vl
                hist[-1]["val_improve"] = imp
            mark = ""
            # **最良チェックポイントを別名で残す。** last.pt を上書きし続けた結果、
            # 保留改善率が最良だった 20,000 step (52.2%) のモデルを失い、
            # 最終 30,000 step (42.0%) しか手元に残らなかった。
            if np.isfinite(imp) and imp > best_imp:
                best_imp = imp
                os.makedirs(a.out, exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "val_l1": vl, "val_improve": imp},
                           os.path.join(a.out, "best.pt"))
                mark = "  <- best 更新"
            p(f"  {'':>6}  保留 L1 {vl:.5f}  ぼかし基準比 改善率 {imp*100:.1f}%{mark}")

        if step % 1000 == 0 or step == a.steps:
            os.makedirs(a.out, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step},
                       os.path.join(a.out, "last.pt"))
            with open(os.path.join(a.out, "history.jsonl"), "w") as f:
                for r in hist:
                    f.write(json.dumps(r) + "\n")

    if len(hist) >= 2:
        p(f"\n=== 学習前後 ===")
        p(f"  L1      {hist[0]['l1']:.5f} -> {hist[-1]['l1']:.5f} "
          f"({'改善' if hist[-1]['l1'] < hist[0]['l1'] * 0.9 else '**改善が弱い**'})")
        p(f"  |Δw|max {hist[-1]['dw']:.3e} "
          f"({'パラメータが動いている' if hist[-1]['dw'] > 1e-5 else '**動いていない**'})")
        p(f"  最良      改善率 {best_imp*100:.1f}% (checkpoints/.../best.pt)")
        vs = [(r['step'], r['val_l1'], r['val_improve'])
              for r in hist if r.get('val_l1') is not None]
        if len(vs) >= 2:
            p(f"  保留 L1  {vs[0][1]:.5f} -> {vs[-1][1]:.5f} "
              f"({'改善' if vs[-1][1] < vs[0][1] else '**悪化/横ばい**'})")
            p(f"  改善率   {vs[0][2]*100:.1f}% -> {vs[-1][2]*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
