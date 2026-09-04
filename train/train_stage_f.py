"""Stage F の学習 — 全解像度直接補完の実現可能性検証 (train_stage_a.py を出発点)。

**目的はモデルの完成ではなく問いに答えること。** Stage A (1/8 低周波専用) は
LaMa に勝てないことが確定した。「全解像度で直接補完すれば LaMa に迫れるのか」を
短時間の学習で見極める。GAN も知覚損失も使わない。実現可能性の一次スクリーニング
であって、勝てるかどうかの最終判断はもっと長い本学習で行う。

`train_stage_a.py` との違い:
  - 低周波化 (`low_target`) をしない。画像・マスクは全解像度のまま使う。
  - Stage F はマスク外もネットワークの生出力を返すので、損失側で
    穴の L1 + 既知領域の L1 (重み0.3) + 勾配損失 (重み0.5) を全解像度で計算する
    (`stage_a.py` の `step_loss`/`grad_loss` と同じ式、SHRINK を掛けないだけ)。
  - 全解像度は 512² で MPS/CPU ともメモリが厳しいので `--crop` でランダムクロップする。

過学習ゲート (`--overfit`) は StageA と同じ理由で必須: 数百 step で loss が
明確に下がらない構成のまま長時間学習を回すと、後で「構造が壊れていた」
だけだったと分かって時間を失う (StageA 開発で4回踏んだ失敗)。

  pixi run python train/train_stage_f.py --overfit 2 --steps 300 --crop 128   # 関門
  pixi run python train/train_stage_f.py --steps 6000
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

from eval.quickval import quick_eval                              # noqa: E402
from nagi_inpaint2.stage_f import StageF                          # noqa: E402
from train.dataset import list_shards, make_loader, split_shards  # noqa: E402


def p(*a):
    print(*a, flush=True)


def random_crop(img: torch.Tensor, mask: torch.Tensor, crop: int):
    """タイル辺 (512) が `--crop` より大きいときだけランダムクロップする。

    全バッチで同じ切り出し位置を使う (StageA 由来の `fixed` タプルもそのまま
    再利用したいのでバッチ単位でシンプルにしておく)。タイルが crop 以下なら
    そのまま返す。
    """
    h, w = img.shape[-2:]
    if h <= crop and w <= crop:
        return img, mask
    ch, cw = min(crop, h), min(crop, w)
    y0 = int(torch.randint(0, h - ch + 1, (1,)))
    x0 = int(torch.randint(0, w - cw + 1, (1,)))
    return (img[..., y0:y0 + ch, x0:x0 + cw],
            mask[..., y0:y0 + ch, x0:x0 + cw])


def grad_loss(a, b):
    """勾配も合わせる (stage_a.py と同じ式)。構造の位置ずれを抑える。"""
    return ((a[:, :, 1:] - a[:, :, :-1]) - (b[:, :, 1:] - b[:, :, :-1])).abs().mean() \
        + ((a[..., 1:] - a[..., :-1]) - (b[..., 1:] - b[..., :-1])).abs().mean()


def step_loss(model, img, mask, lam_grad: float, lam_known: float = 0.3):
    """穴の L1 + 勾配 + 既知領域の L1。全解像度版 (`stage_a.step_loss` の SHRINK 抜き)。"""
    pred = model(img, mask)
    m = mask.expand_as(pred)
    if m.sum() < 1:
        return None, None
    l1 = ((pred - img).abs() * m).sum() / (m.sum() + 1e-6)
    k = 1.0 - m
    l1k = ((pred - img).abs() * k).sum() / (k.sum() + 1e-6)
    lg = grad_loss(pred * mask, img * mask)
    return l1 + lam_grad * lg + lam_known * l1k, l1


@torch.no_grad()
def validate(model, it, n: int, device: str, crop: int):
    """保留 shard での L1 と、ぼかし基準に対する改善率 (`stage_a.validate` の全解像度版)。"""
    model.eval()
    tot_s = tot_b = 0.0
    c = 0
    for _ in range(n):
        b = next(it)
        img, mask = b["image"].to(device), b["mask"].to(device)
        img, mask = random_crop(img, mask, crop)
        m = mask.expand(-1, 3, -1, -1)
        if m.sum() < 1:
            continue
        pred = model(img, mask)
        blur = F.avg_pool2d(F.pad(img * (1 - mask), (4, 4, 4, 4), mode="replicate"),
                            9, stride=1) / (
            F.avg_pool2d(F.pad(1 - mask, (4, 4, 4, 4), mode="replicate"), 9, stride=1) + 1e-6)
        tot_s += float(((pred - img).abs() * m).sum() / m.sum())
        tot_b += float(((blur - img).abs() * m).sum() / m.sum())
        c += 1
    model.train()
    if c == 0:
        return float("nan"), float("nan")
    return tot_s / c, 1.0 - (tot_s / c) / max(tot_b / c, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop", type=int, default=256,
                    help="全解像度 (512) がメモリに乗らない場合のランダムクロップ辺")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam-grad", type=float, default=0.5)
    ap.add_argument("--base", type=int, default=32)
    # マスク種別を固定しない (StageA と同じ理由: blob だけでは分布が狭く汎化しない)
    ap.add_argument("--mask-kind", default=None)
    ap.add_argument("--out", default="checkpoints/stage_f")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=12)
    ap.add_argument("--quickval-cases", type=int, default=10,
                    help="eval/quickval.py の quick_eval を val-every 毎に呼ぶ"
                         "ケース数。0 で無効。best.pt の選択はこちらの psnr を"
                         "使う (validate() の代理指標では選ばない — 学習時の"
                         "検証と最終評価の乖離を防ぐため。SPEC/このファイルの"
                         "docstring 参照)")
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

    model = StageF(base=a.base).to(a.device)
    n_par = sum(q.numel() for q in model.parameters())
    p(f"StageF params: {n_par:,}  (全解像度で動作、crop={a.crop})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    w0 = {k: v.detach().clone() for k, v in model.named_parameters()}

    fixed = None
    if a.overfit:
        b = next(it)
        img, mask = b["image"][:a.overfit].to(a.device), b["mask"][:a.overfit].to(a.device)
        fixed = random_crop(img, mask, a.crop)
        p(f"過学習テスト: 固定 {a.overfit} タイル (crop={a.crop})")

    hist = []
    best_imp = -1e9          # 保留の改善率 (参考値、best.pt の選択には使わない)
    best_qv_psnr = -1e9      # quick_eval の psnr が最良のときのモデルを別に残す
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        if fixed is not None:
            img, mask = fixed
        else:
            b = next(it)
            img, mask = b["image"].to(a.device), b["mask"].to(a.device)
            img, mask = random_crop(img, mask, a.crop)
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
                # **buffers を含めない。** named_parameters() のみで |Δw|max を計算する
                # (StageA で BatchNorm の running_var 暴走を誤読した教訓)。
                dw = max(float((q - w0[k]).abs().max())
                         for k, q in model.named_parameters())
            el = time.perf_counter() - t0
            rec = dict(step=step, loss=float(loss.detach()), l1=float(l1.detach()),
                       dw=dw, lr=float(sched.get_last_lr()[0]), sec_per_step=el / step)
            hist.append(rec)
            p(f"  {step:>6}/{a.steps}  loss {rec['loss']:.5f}  L1 {rec['l1']:.5f}  "
              f"|Δw|max {dw:.3e}  {rec['sec_per_step']*1000:.0f}ms/step")

        if step % a.val_every == 0 or step == a.steps:
            vl, imp = validate(model, vit, a.val_batches, a.device, a.crop)
            if hist:
                hist[-1]["val_l1"] = vl
                hist[-1]["val_improve"] = imp

            # **quick_eval で最終評価と同じ経路・同じ指標を測る。**
            # `validate()` の L1/改善率は自前の代理指標であり、実際に
            # eval/benchmark.py が測る PSNR と乖離しうる (このセッションで
            # 4回踏んだ失敗の一つ: WTA 学習の検証は grain 1.021 を報告したが
            # 人物マスクでは 0.098 だった)。best.pt の選択は必ずこちらの
            # psnr を使う。
            qv = None
            if a.quickval_cases > 0:
                model.eval()

                def query_fn(x, mk):
                    return torch.where(mk > 0.5, model(x, mk), x)

                qv = quick_eval(query_fn, n_cases=a.quickval_cases,
                                device=a.device, shard_dir=a.shards)
                model.train()
                if hist:
                    hist[-1]["quickval"] = qv
                p(f"  [quickval] grain={qv['grain']:.4f}  psd={qv['psd']:.4f}  "
                  f"psnr={qv['psnr']:.2f}")

            mark = ""
            # 最良チェックポイントを別名で残す (StageA で last.pt 上書きにより
            # 最良モデルを失った教訓と同じ)。**判定は quick_eval の psnr のみ**
            # (validate() の改善率では選ばない)。
            if qv is not None and np.isfinite(qv["psnr"]) and qv["psnr"] > best_qv_psnr:
                best_qv_psnr = qv["psnr"]
                os.makedirs(a.out, exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "val_l1": vl, "val_improve": imp, "quickval": qv},
                           os.path.join(a.out, "best.pt"))
                mark = "  <- best 更新 (quickval psnr が最良)"
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
        p(f"  最良 (参考、改善率) {best_imp*100:.1f}%")
        p(f"  最良 (quickval psnr) {best_qv_psnr:.2f} (checkpoints/.../best.pt)")
        vs = [(r['step'], r['val_l1'], r['val_improve'])
              for r in hist if r.get('val_l1') is not None]
        if len(vs) >= 2:
            p(f"  保留 L1  {vs[0][1]:.5f} -> {vs[-1][1]:.5f} "
              f"({'改善' if vs[-1][1] < vs[0][1] else '**悪化/横ばい**'})")
            p(f"  改善率   {vs[0][2]*100:.1f}% -> {vs[-1][2]*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
