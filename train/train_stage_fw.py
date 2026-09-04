"""StageFW (勝者総取り学習) の学習 — `train_stage_f.py` を出発点にする。

**なぜ WTA か (背景)**

現行の `StageF` は L1 損失で学習しており、平均値回帰でぼける。実測: 出力の
grain (マスク内外の σ 比、理想 1.0) が **0.109**。LaMa は 0.943。人物マスク
の出力を目視すると、穴の位置にぼけた塊が残り、明らかに破綻している。

敵対的損失 (GAN) で直そうとして2回失敗した (grain 0.322 → 0.339 と動かず、
長く回すと崩壊して勝率 34/60 → 10/60)。batch 4・M1 では GAN を安定させられ
なかった。

L1 がぼけるのは「多峰性の分布に対して平均を出す」から。WTA は K 個の仮説を
出し、**最良の1つにだけ勾配を流す**ので、平均を作る動機が消える。学習は L1
と同程度に安定なはずで、識別器も min-max も無い。これがこの実験の仮説であり、
`grain` 比が 1.0 に近づくかどうかで検証する。

**緩和つき WTA を使う理由**: 純粋な WTA (勝者だけに勾配) だと一度も勝てない
ヘッドに勾配が流れず「死ぬ」。少量の勾配 (`--eps`) を全ヘッドに流して防ぐ。

過学習ゲート (`--overfit`) は `StageA`/`StageF` と同じ理由で必須:

  .pixi/envs/default/bin/python train/train_stage_fw.py --overfit 2 --steps 60 \\
      --crop 128 --device cpu --log-every 10 --out /tmp/fw_gate
  .pixi/envs/default/bin/python train/train_stage_fw.py --steps 6000
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

from eval.quickval import quick_eval                               # noqa: E402
from nagi_inpaint2.stage_fw import StageFW                        # noqa: E402
from train.dataset import list_shards, make_loader, split_shards  # noqa: E402
from train.train_stage_f import random_crop                       # noqa: E402


def p(*a):
    print(*a, flush=True)


def grad_loss_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`train_stage_f.grad_loss` と同じ量 (勾配の食い違いの絶対値平均) を計算
    するが、先頭次元 (サンプル×仮説) をスカラーに潰さず残す。

    a, b: (..., C, H, W)。戻り値: (...,) — 末尾3次元 (C,H,W) だけを潰した平均。
    """
    dh = ((a[..., 1:, :] - a[..., :-1, :]) - (b[..., 1:, :] - b[..., :-1, :])).abs()
    dw = ((a[..., :, 1:] - a[..., :, :-1]) - (b[..., :, 1:] - b[..., :, :-1])).abs()
    return dh.flatten(start_dim=-3).mean(dim=-1) + dw.flatten(start_dim=-3).mean(dim=-1)


def per_hypothesis_loss(pred: torch.Tensor, img: torch.Tensor, mask: torch.Tensor,
                        lam_grad: float, lam_known: float = 0.3):
    """K 個の仮説それぞれについて `train_stage_f.step_loss` と同じ式
    (穴の L1 + `lam_known`*既知領域 L1 + `lam_grad`*勾配損失) を計算する。

    **画素単位ではなくサンプル×仮説ごとのスカラーにする。** 画素単位で最小を
    取ると、1枚の仮説の中で画素ごとに別の仮説を継ぎ接ぎすることになり、
    「1つの仮説が1つの一貫した補完」という WTA の前提が崩れる。

    pred: (N,K,3,H,W)  img: (N,3,H,W)  mask: (N,1,H,W)
    戻り値: (N,K) の損失、または穴が空 (マスク全ゼロ) なら None。
    """
    n, k = pred.shape[0], pred.shape[1]
    m3 = mask.expand(-1, 3, -1, -1)                      # (N,3,H,W)
    if m3.sum() < 1:
        return None
    m = m3.unsqueeze(1)                                   # (N,1,3,H,W) -> K へブロードキャスト
    img_e = img.unsqueeze(1)                               # (N,1,3,H,W)
    diff = (pred - img_e).abs()                             # (N,K,3,H,W)
    hole = (diff * m).sum(dim=(2, 3, 4)) / (m.sum(dim=(2, 3, 4)) + 1e-6)
    known = 1.0 - m
    l1k = (diff * known).sum(dim=(2, 3, 4)) / (known.sum(dim=(2, 3, 4)) + 1e-6)
    pm = pred * mask.unsqueeze(1)                            # マスク内だけで勾配損失
    im = (img * mask).unsqueeze(1).expand(-1, k, -1, -1, -1)
    lg = grad_loss_per_sample(pm, im)
    return hole + lam_known * l1k + lam_grad * lg


def relaxed_wta(per_k: torch.Tensor, eps: float = 0.05):
    """緩和つき WTA。per_k: (...,K)。

    最小の損失を持つ仮説を「勝者」とし、重み `1-eps` を、残りに
    `eps/(K-1)` を配る。**理由**: 純粋な WTA (勝者だけ勾配) だと一度も勝てない
    ヘッドに勾配が流れず「死ぬ」(そのヘッドは永遠に劣ったまま、多様性が
    育たない)。少量の勾配を全ヘッドに流して死を防ぐ。

    戻り値: (loss, winner_idx, weights)。`loss`/`winner_idx` は per_k から
    末尾次元 (K) を消した形。
    """
    k = per_k.shape[-1]
    winner = per_k.argmin(dim=-1)
    w = torch.full_like(per_k, eps / max(k - 1, 1))
    w.scatter_(-1, winner.unsqueeze(-1), 1.0 - eps)
    loss = (w * per_k).sum(dim=-1)
    return loss, winner, w


def step_loss_wta(model, img, mask, lam_grad: float, eps: float, lam_known: float = 0.3):
    """WTA 損失一式をまとめて計算する (`train_stage_f.step_loss` の WTA 版)。

    戻り値: (loss, info)。`loss` は backward 対象のスカラー。`info` は
    監視用の dict (`winner`: (N,) 勝者index, `win_loss`: 勝者損失の平均,
    `std_k`: サンプルごとの K 損失の標準偏差の平均 (仮説の多様性の指標),
    `per_k`: (N,K) 生の仮説別損失)。穴が空なら (None, None)。
    """
    pred = model(img, mask)                               # (N,K,3,H,W)
    per_k = per_hypothesis_loss(pred, img, mask, lam_grad, lam_known)
    if per_k is None:
        return None, None
    loss_per_sample, winner, _w = relaxed_wta(per_k, eps)
    loss = loss_per_sample.mean()
    with torch.no_grad():
        win_loss = per_k.gather(1, winner.unsqueeze(1)).squeeze(1).mean()
        std_k = per_k.std(dim=1).mean() if per_k.shape[1] > 1 else torch.zeros(())
    info = dict(winner=winner, win_loss=win_loss, std_k=std_k, per_k=per_k, pred=pred)
    return loss, info


# --- 保留 shard での評価: grain 比 (torch 版) ---------------------------------
#
# **`F.interpolate(..., mode="area")` は使わない。** MPS で落ちる (実際に学習が
# 落ちた事例がある)。ここでは interpolate 自体を使わず、シフトと max/avg_pool
# だけで実装する。

def _dilate4_torch(mask: torch.Tensor, r: int) -> torch.Tensor:
    """4近傍膨張。`eval/metrics.py` の `_dilate` を torch (0/1 float) で書き直したもの。"""
    m = mask
    for _ in range(r):
        o = m.clone()
        o[..., 1:, :] = torch.maximum(o[..., 1:, :], m[..., :-1, :])
        o[..., :-1, :] = torch.maximum(o[..., :-1, :], m[..., 1:, :])
        o[..., :, 1:] = torch.maximum(o[..., :, 1:], m[..., :, :-1])
        o[..., :, :-1] = torch.maximum(o[..., :, :-1], m[..., :, 1:])
        m = o
    return m


def _highpass_torch(x: torch.Tensor) -> torch.Tensor:
    """3x3 箱平均を引いた最細ディテール帯 (`eval/metrics.py` の `_highpass` の torch 版)。"""
    return x - F.avg_pool2d(x, 3, stride=1, padding=1)


def grain_ratio_torch(out: torch.Tensor, mask: torch.Tensor, annulus: int = 32) -> float:
    """マスク内外の高周波 σ 比 (`eval/metrics.py` の `grain_ratio` の torch 版)。

    理想は 1.0 (マスク内外で粒状が同じ)。**grain がこの実験の主目的**なので
    必ず出す。現行の L1 版 StageF は 0.109。out: (1,3,H,W)、mask: (1,1,H,W)。
    σ は知覚空間のまま計算する (仕様どおり)。
    """
    g = out.mean(dim=1, keepdim=True)
    hp = _highpass_torch(g)
    m = (mask > 0.5).float()
    ann = _dilate4_torch(m, annulus) * (1.0 - m)
    # 境界の勾配を拾わないよう、マスクは少し内側に寄せる
    inner = m * (1.0 - _dilate4_torch(1.0 - m, 3))
    ann_b, inner_b = ann > 0.5, inner > 0.5
    if int(inner_b.sum()) < 64 or int(ann_b.sum()) < 64:
        return float("nan")
    so = float(hp[ann_b].std())
    si = float(hp[inner_b].std())
    return si / so if so > 1e-12 else float("nan")


@torch.no_grad()
def validate(model, it, n: int, device: str, crop: int, eps: float, lam_grad: float):
    """保留 shard で、勝者選択後の L1 と grain 比を測る。

    **grain がこの実験の主目的。** L1 だけを見ていた `StageF` は grain 0.109
    までぼけていたのに L1 自体は改善して見えていた。数値の見た目に騙されない
    よう必ず両方報告する。勝者選択は per_k (訓練損失と同じ式) の argmin で行う
    (GT を使うが、これは学習目的そのものの評価であって `eval/benchmark.py` の
    "oracle" 診断とは別物)。
    """
    model.eval()
    tot_l1 = tot_grain = 0.0
    n_l1 = n_grain = 0
    win_counts = None
    for _ in range(n):
        b = next(it)
        img, mask = b["image"].to(device), b["mask"].to(device)
        img, mask = random_crop(img, mask, crop)
        if mask.sum() < 1:
            continue
        pred = model(img, mask)                            # (N,K,3,H,W)
        per_k = per_hypothesis_loss(pred, img, mask, lam_grad)
        if per_k is None:
            continue
        _, winner, _ = relaxed_wta(per_k, eps)
        k = pred.shape[1]
        if win_counts is None:
            win_counts = torch.zeros(k)
        win_counts += torch.bincount(winner.cpu(), minlength=k).float()
        idx = winner.view(-1, 1, 1, 1, 1).expand(-1, 1, *pred.shape[2:])
        chosen = pred.gather(1, idx).squeeze(1)              # (N,3,H,W)
        m3 = mask.expand(-1, 3, -1, -1)
        tot_l1 += float(((chosen - img).abs() * m3).sum() / (m3.sum() + 1e-6))
        n_l1 += 1
        full = torch.where(mask > 0.5, chosen, img)
        for i in range(full.shape[0]):
            g = grain_ratio_torch(full[i:i + 1], mask[i:i + 1])
            if np.isfinite(g):
                tot_grain += g
                n_grain += 1
    model.train()
    val_l1 = tot_l1 / n_l1 if n_l1 else float("nan")
    val_grain = tot_grain / n_grain if n_grain else float("nan")
    if win_counts is not None and float(win_counts.sum()) > 0:
        wr = (win_counts / win_counts.sum()).numpy()
    else:
        wr = None
    return val_l1, val_grain, wr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256,
                    help="全解像度 (512) がメモリに乗らない場合のランダムクロップ辺")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-grad", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=4, help="仮説の数")
    ap.add_argument("--eps", type=float, default=0.05,
                    help="緩和つき WTA: 敗者に配る勾配の総量。0 に近いほど純粋な WTA")
    ap.add_argument("--base", type=int, default=96,
                    help="--init が無いときのフォールバック (--init があれば checkpoint "
                         "の out.weight の形から復元する)")
    ap.add_argument("--init", default="checkpoints/stage_f96/last.pt",
                    help="StageF の checkpoint から backbone を読み込む。無ければランダム初期化")
    # マスク種別を固定しない (StageA/StageF と同じ理由: blob だけでは分布が狭く汎化しない)
    ap.add_argument("--mask-kind", default=None)
    ap.add_argument("--out", default="checkpoints/stage_fw")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=12)
    ap.add_argument("--quickval-cases", type=int, default=10,
                    help="eval/quickval.py の quick_eval を val-every 毎に呼ぶ"
                         "ケース数。0 で無効。best.pt の選択はこちらの grain"
                         "(1.0 に最も近いとき) を使う (validate() の混合マスク"
                         "grain では選ばない — 細い穴では grain が測れず盲点に"
                         "なる。このファイル冒頭の docstring 参照)")
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

    base = a.base
    init_sd = None
    if a.init and os.path.exists(a.init):
        ck = torch.load(a.init, map_location=a.device)
        base = int(ck.get("base") or ck["model"]["out.weight"].shape[1])
        init_sd = ck["model"]
        p(f"  backbone 初期化: {a.init} (base={base} を checkpoint から復元, "
          f"step {ck.get('step')})")
    else:
        p(f"  backbone 初期化なし (--init {a.init} が見つからない): "
          f"base={base} でランダム初期化")

    model = StageFW(base=base, k=a.k).to(a.device)
    if init_sd is not None:
        model.load_backbone(init_sd)
    n_par = sum(q.numel() for q in model.parameters())
    p(f"StageFW params: {n_par:,}  (base={base}, k={a.k}, crop={a.crop}, eps={a.eps})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    w0 = {name: v.detach().clone() for name, v in model.named_parameters()}

    fixed = None
    if a.overfit:
        b = next(it)
        img, mask = b["image"][:a.overfit].to(a.device), b["mask"][:a.overfit].to(a.device)
        fixed = random_crop(img, mask, a.crop)
        p(f"過学習テスト: 固定 {a.overfit} タイル (crop={a.crop})")

    hist = []
    best_diff = 1e9          # quick_eval の |grain - 1.0| が最小のときのモデルを別に残す
    win_acc = torch.zeros(a.k)   # 直近ログ区間の勝者カウント (偏りの検出用)
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        if fixed is not None:
            img, mask = fixed
        else:
            b = next(it)
            img, mask = b["image"].to(a.device), b["mask"].to(a.device)
            img, mask = random_crop(img, mask, a.crop)
        loss, info = step_loss_wta(model, img, mask, a.lam_grad, a.eps)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        win_acc += torch.bincount(info["winner"].cpu(), minlength=a.k).float()

        if step % a.log_every == 0:
            with torch.no_grad():
                # **buffers を含めない。** named_parameters() のみで |Δw|max を計算する
                # (StageA/StageF で BatchNorm の running_var 暴走を誤読した教訓)。
                dw = max(float((q - w0[name]).abs().max())
                         for name, q in model.named_parameters())
            wr = ((win_acc / win_acc.sum()).numpy()
                  if float(win_acc.sum()) > 0 else np.zeros(a.k))
            el = time.perf_counter() - t0
            rec = dict(step=step, loss=float(loss.detach()),
                      win_loss=float(info["win_loss"]), std_k=float(info["std_k"]),
                      dw=dw, lr=float(sched.get_last_lr()[0]), win_rate=wr.tolist(),
                      sec_per_step=el / step)
            hist.append(rec)
            p(f"  {step:>6}/{a.steps}  loss {rec['loss']:.5f}  "
              f"winner_loss {rec['win_loss']:.5f}  std_k {rec['std_k']:.5f}  "
              f"|Δw|max {dw:.3e}  勝率 {np.array2string(wr, precision=2, suppress_small=True)}  "
              f"{rec['sec_per_step']*1000:.0f}ms/step")
            win_acc.zero_()

        if step % a.val_every == 0 or step == a.steps:
            vl, vg, vwr = validate(model, vit, a.val_batches, a.device, a.crop,
                                   a.eps, a.lam_grad)
            if hist:
                hist[-1]["val_l1"] = vl
                hist[-1]["val_grain"] = vg
                hist[-1]["val_win_rate"] = vwr.tolist() if vwr is not None else None

            # **quick_eval で最終評価と同じ経路・同じ指標を測る。**
            # `validate()` の保留 grain は学習に使う混合マスク (電線などの
            # 細いものを含む) で測っており、細い穴では grain_ratio が侵食で
            # 内部を失って測れず、平均から静かに除外される (このファイル冒頭
            # の docstring、および tests/test_metrics_sanity.py の
            # test_grain_is_blind_on_thin_masks 参照)。実際に grain 1.021 を
            # 報告しながら人物マスクでは 0.098 だった失敗の原因がこれ。
            # best.pt の選択は必ず quick_eval (人物サイズマスク) の grain を使う。
            qv = None
            if a.quickval_cases > 0:
                model.eval()

                def query_fn(x, mk):
                    return torch.where(mk > 0.5, model(x, mk)[:, 0], x)

                qv = quick_eval(query_fn, n_cases=a.quickval_cases,
                                device=a.device, shard_dir=a.shards)
                model.train()
                if hist:
                    hist[-1]["quickval"] = qv
                p(f"  [quickval] grain={qv['grain']:.4f}  psd={qv['psd']:.4f}  "
                  f"psnr={qv['psnr']:.2f}")

            mark = ""
            # 最良チェックポイントを別名で残す (StageA/StageF で last.pt 上書きにより
            # 最良モデルを失った教訓と同じ)。**判定は quick_eval の grain のみ**
            # (validate() の混合マスク grain では選ばない)。
            if qv is not None and np.isfinite(qv["grain"]) and abs(qv["grain"] - 1.0) < best_diff:
                best_diff = abs(qv["grain"] - 1.0)
                os.makedirs(a.out, exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                           "base": base, "k": a.k, "val_l1": vl, "val_grain": vg,
                           "quickval": qv},
                          os.path.join(a.out, "best.pt"))
                mark = "  <- best 更新 (quickval grain が 1.0 に最も近い)"
            vwr_s = (np.array2string(vwr, precision=2, suppress_small=True)
                    if vwr is not None else "n/a")
            p(f"  {'':>6}  保留 L1 {vl:.5f}  保留 grain {vg:.4f} (理想 1.0)  "
              f"保留勝率 {vwr_s}{mark}")

        if step % 500 == 0 or step == a.steps:
            os.makedirs(a.out, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step,
                       "base": base, "k": a.k},
                      os.path.join(a.out, "last.pt"))
            with open(os.path.join(a.out, "history.jsonl"), "w") as f:
                for r in hist:
                    f.write(json.dumps(r) + "\n")

    if len(hist) >= 2:
        p(f"\n=== 学習前後 ===")
        p(f"  loss      {hist[0]['loss']:.5f} -> {hist[-1]['loss']:.5f} "
          f"({'改善' if hist[-1]['loss'] < hist[0]['loss'] * 0.9 else '**改善が弱い**'})")
        p(f"  |Δw|max   {hist[-1]['dw']:.3e} "
          f"({'パラメータが動いている' if hist[-1]['dw'] > 1e-5 else '**動いていない**'})")
        last_wr = hist[-1].get("win_rate")
        if last_wr:
            mx = max(last_wr)
            p(f"  勝率       {np.array2string(np.array(last_wr), precision=2)} "
              f"({'**偏っている (WTA が機能していない可能性)**' if mx > 0.7 else '概ね分散'})")
        vs = [(r["step"], r["val_l1"], r["val_grain"])
              for r in hist if r.get("val_l1") is not None]
        if len(vs) >= 2:
            p(f"  保留 L1    {vs[0][1]:.5f} -> {vs[-1][1]:.5f}")
            p(f"  保留 grain {vs[0][2]:.4f} -> {vs[-1][2]:.4f} (理想 1.0)")
        p(f"  最良 (quickval) |grain-1| = {best_diff:.4f} (checkpoints/.../best.pt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
