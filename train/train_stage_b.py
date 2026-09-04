"""Stage B の学習 (SPEC §7)。

**この学習が答えるべき問い**

初期状態では top-k の重みが均等 (エントロピー 1.000) になっている。
FeatureNet がランダム初期化で、離れた候補同士のコスト差が付かないため。
損失は「gt に近い候補の重みを上げる」方向に働くので、
**学習でコスト差が開き、ブレンドが選択的になるか**が本質的な問い。

  - 開けば: 非パラメトリック合成が artifact を抑えられる = 設計が成立する
  - 開かなければ: 候補の作り方か特徴の設計を見直す必要がある

そのため損失だけでなく**重みエントロピーを毎回記録する**。

損失 (§7.1):
    L1(合成, gt)_mask + λ·spectral(合成, gt)_mask
GAN は使わない (§3.2)。高周波は実画素から供給される。

  pixi run python train/train_stage_b.py --steps 2000
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

from nagi_inpaint2.stage_b import StageB, init_offsets_topk    # noqa: E402
from train.dataset import list_shards, make_loader, split_shards  # noqa: E402


def p(*a):
    print(*a, flush=True)


def spectral_loss(a, b, mask):
    """高周波の欠落を罰する。マスク領域の振幅スペクトルを合わせる。"""
    m = mask.expand_as(a)
    fa = torch.fft.rfft2(a * m).abs()
    fb = torch.fft.rfft2(b * m).abs()
    return (torch.log1p(fa) - torch.log1p(fb)).abs().mean()


def anneal_tau(step: int, total: int, hi: float = 1.0, lo: float = 0.05) -> float:
    """τ を hi -> lo へ指数的に焼き鈍す。

    学習可能にすると L1 が τ を増やして均等平均へ倒す (実測)。
    soft ブレンドは微分可能性のための手段で、収束先は hard selection。
    lo を 0 にしないのは、softmax が完全に飽和すると
    FeatureNet への勾配が消えるため。
    """
    r = min(step / max(total, 1), 1.0)
    return float(hi * (lo / hi) ** r)


def weight_entropy(model, cost_k, mask, tau):
    """top-k 重みのエントロピー。0=argmin, 1=均等。"""
    C = cost_k - cost_k.min(dim=1, keepdim=True).values
    C = C / (C.mean(dim=1, keepdim=True) + 1e-8)
    w = torch.softmax(-C / max(tau, 1e-3), dim=1)
    e = -(w * torch.log(w + 1e-9)).sum(1) / np.log(w.shape[1])
    sel = mask[:, 0] > 0.5
    return float(e[sel].median()) if sel.any() else float("nan")


@torch.no_grad()
def validate(model, loader_it, n_batches: int, tau: float, device: str):
    """**固定 τ** で検証する。

    学習中の loss は τ の焼き鈍しに支配されてしまう
    (平均をやめて実パッチ1枚を選べば L1 は必ず上がる) ため、
    学習が実際に品質を上げているかは固定 τ でしか測れない。
    """
    model.eval()
    tot, n = 0.0, 0
    for _ in range(n_batches):
        b = next(loader_it)
        img = b["image"].to(device)
        mask = b["mask"].to(device)
        oy, ox, ck = init_offsets_topk(img, mask, model.f)
        comp, _ = model(img, mask, oy, ox, ck, tau=tau)
        sel = mask.expand_as(img) > 0.5
        if sel.any():
            mse = ((comp - img) ** 2)[sel].mean()
            tot += float(10 * torch.log10(1.0 / mse.clamp_min(1e-12)))
            n += 1
    model.train()
    return tot / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    # 平均化の利得を相殺する。0.1 では L1 に押し負けた (実測)
    ap.add_argument("--lam-spectral", type=float, default=1.0)
    ap.add_argument("--mask-kind", default="blob",
                    help="LaMa の失敗条件は blob × 高周波テクスチャ")
    ap.add_argument("--out", default="checkpoints/stage_b")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tr_shards, va_shards = split_shards(list_shards(a.shards))
    dl = make_loader(tr_shards, batch_size=a.batch, workers=a.workers,
                     mask_kind=a.mask_kind)
    it = iter(dl)
    vdl = make_loader(va_shards, batch_size=a.batch, workers=1,
                      mask_kind=a.mask_kind, seed=999)
    vit = iter(vdl)

    model = StageB().to(a.device)
    p(f"StageB params: {sum(q.numel() for q in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    hist = []
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        b = next(it)
        img = b["image"].to(a.device)
        mask = b["mask"].to(a.device)
        valid = b["valid"].to(a.device)

        with torch.no_grad():
            oy, ox, ck = init_offsets_topk(img, mask, model.f)

        tau = anneal_tau(step, a.steps)
        comp, conf = model(img, mask, oy, ox, ck, tau=tau)
        # 飽和画素 (highlight_mode=5 の再構成) は実測値ではないので損失から外す (§6.0)
        w = mask * valid
        l1 = ((comp - img).abs() * w).sum() / (w.sum() * 3 + 1e-6)
        ls = spectral_loss(comp, img, mask)
        loss = l1 + a.lam_spectral * ls

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % a.log_every == 0:
            ent = weight_entropy(model, ck, mask, tau)
            el = time.perf_counter() - t0
            rec = dict(step=step, loss=float(loss.detach()),
                       l1=float(l1.detach()), spectral=float(ls.detach()),
                       entropy=ent, tau=tau,
                       conf=float(conf[mask > 0.5].median()) if (mask > 0.5).any() else 0.0,
                       sec_per_step=el / step)
            hist.append(rec)
            p(f"  {step:>6}/{a.steps}  loss {rec['loss']:.5f} "
              f"(L1 {rec['l1']:.5f} spec {rec['spectral']:.5f})  "
              f"重みent {ent:.3f}  τ {rec['tau']:.5f}  conf {rec['conf']:.3f}  "
              f"{rec['sec_per_step']:.2f}s/step")

        if step % a.val_every == 0 or step == a.steps:
            v = validate(model, vit, a.val_batches, 0.05, a.device)
            hist[-1]["val_psnr"] = v if hist else None
            p(f"  {'':>6}  検証 PSNR (τ=0.05 固定) {v:.2f} dB")

        if step % 500 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "step": step},
                       os.path.join(a.out, "last.pt"))
            with open(os.path.join(a.out, "history.jsonl"), "w") as f:
                for r in hist:
                    f.write(json.dumps(r) + "\n")

    if len(hist) >= 2:
        p(f"\n=== 学習前後 ===")
        p(f"  loss     {hist[0]['loss']:.5f} -> {hist[-1]['loss']:.5f}")
        p(f"  重みent  {hist[0]['entropy']:.3f} -> {hist[-1]['entropy']:.3f} "
          f"({'選択的になった' if hist[-1]['entropy'] < hist[0]['entropy'] - 0.02 else '変化なし'})")
        p(f"  τ(焼鈍)  {hist[0]['tau']:.5f} -> {hist[-1]['tau']:.5f}")
        vs = [(r['step'], r['val_psnr']) for r in hist if r.get('val_psnr')]
        if len(vs) >= 2:
            p(f"  検証PSNR {vs[0][1]:.2f} -> {vs[-1][1]:.2f} dB "
              f"({'改善' if vs[-1][1] > vs[0][1] else '悪化/横ばい'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
