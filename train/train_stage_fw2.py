"""StageFW2 (デコーダ分岐 WTA) の学習 — `train/train_stage_fw.py` を出発点にする。

**効く理屈と、効かない条件**

狙い: L1 損失は多峰性の分布に対して平均を出すためぼける (実測: grain 0.109、
LaMa 0.943、理想 1.0)。K 個の仮説を出して**最良の1つにだけ勾配を流す**
(Winner-Take-All) と、平均を作る動機が消える。GAN と違い識別器も min-max も
無いので学習は安定する (GAN は2回試して崩壊した)。

**前回 (`StageFW`) の失敗**: 出力層だけを K 個にした結果、4つのヘッドが
バックボーン全体を共有し、同じ特徴の線形読み出しの違いしか表現できなかった。
仮説がほぼ同一になり、**正解を見て最良を選ぶ神託選択ですら first 選択を
上回らなかった** (PSNR 21.29 vs 21.43)。仮説間の損失の標準偏差も損失の約10%
しかなかった。

**検出方法**: 「神託選択が first 選択を明確に上回るか」。上回らなければ
多様性が無く、WTA は機能していない。**この指標を毎回の検証で必ず出す**
(`[多様性] oracle-first psnr=... lpips=...` の行)。`StageFW2`
(`nagi_inpaint2/stage_fw2.py`) はデコーダ以降を K 本の独立した重みの枝にし、
各枝の入力に枝ごとの学習可能な埋め込みを加算することで、分岐を深くして
この問題に対処する。効いたかどうかは学習結果 (このログの多様性指標) で
判断する。

**このファイルで `train_stage_fw.py` から変えた点**

  1. モデルを `StageFW2` にする。WTA の損失・緩和・勝率表示 (`per_hypothesis_loss`,
     `relaxed_wta`, `step_loss_wta`, `validate` 等) は `train_stage_fw.py` から
     そのまま import して再利用する (モデルの内部構造に依存しない汎用コードのため)。
  2. `eval/quickval.py` の `quick_eval` を使った検証を必須にする (SPEC R11)。
     **best.pt の選択は `quick_eval` の `lpips` (`eval.perceptual.Perceptual`
     による知覚距離、小さいほど良い) が最小のときとする。** `train_stage_fw.py`
     は grain で選んでいたが、grain は細いマスクで盲目になる
     (`tests/test_metrics_sanity.py` の `test_grain_is_blind_on_thin_masks`)。
     lpips は陰性対照テスト済み (`R12`、ぼかし検出 0.0019 -> 0.0035)。
  3. **多様性の監視を追加する (最重要)**。毎回の検証で `first` 選択と
     `oracle` 選択それぞれの quick_eval (psnr, lpips) を出し、その差
     (`oracle - first`) をログに出す。**oracle は正解 (GT) を見て K 個の
     仮説から最良を選ぶ診断専用の指標であり、製品経路ではない。** 差が
     0 に近ければ分岐が仮説の多様性を生んでいないことを意味する。

  過学習ゲート (`StageA`/`StageF`/`StageFW` と同じ理由で必須):

    .pixi/envs/default/bin/python train/train_stage_fw2.py \\
        --init checkpoints/stage_f96/last.pt --overfit 2 --steps 40 \\
        --crop 128 --device cpu --log-every 10 --quickval-cases 0 \\
        --out /tmp/fw2_gate
    .pixi/envs/default/bin/python train/train_stage_fw2.py --steps 6000   # 本学習 (mps)
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

from eval.quickval import quick_eval                               # noqa: E402
from nagi_inpaint2.stage_fw import select_hypothesis               # noqa: E402
from nagi_inpaint2.stage_fw2 import StageFW2                       # noqa: E402
from train.dataset import list_shards, make_loader, split_shards   # noqa: E402
from train.train_stage_f import random_crop                        # noqa: E402
from train.train_stage_fw import (per_hypothesis_loss, relaxed_wta,  # noqa: E402
                                  step_loss_wta, validate)


def p(*a):
    print(*a, flush=True)


# --- quick_eval 用の query_fn ------------------------------------------------
#
# `eval/quickval.py` の `query_fn(x, mk)` は x に**穴埋め前の GT 画像**を渡す
# (StageF/StageFW2 が内部で `img*(1-mask)` により穴を自分でゼロ埋めするため)。
# なので oracle 選択もこの x を GT として使える (`select_hypothesis` の
# `hyp="oracle"` 経路と全く同じ関数を再利用する)。

def make_query_fn(model: StageFW2, hyp: str):
    """`hyp` ("first" | "oracle") で仮説を選ぶ quick_eval 用クエリを作る。

    **oracle は診断専用。** GT (`x`) を見て K 個の仮説から最良を選ぶので、
    製品では使えない上限測定であり、多様性 (分岐が機能しているか) を測る
    ためだけに使う。
    """
    def query_fn(x: torch.Tensor, mk: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds = model(x, mk)                     # (N,K,3,H,W)
            chosen = select_hypothesis(preds, x, mk, hyp)
        return torch.where(mk > 0.5, chosen, x)
    return query_fn


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
                         "の base から復元する)")
    ap.add_argument("--init", default="checkpoints/stage_f96/last.pt",
                    help="StageF の checkpoint から共有部分 (stem/enc/mid) と "
                         "初期デコーダ重みを読み込む。無ければランダム初期化")
    ap.add_argument("--mask-kind", default=None)
    ap.add_argument("--out", default="checkpoints/stage_fw2")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=12)
    ap.add_argument("--quickval-cases", type=int, default=10,
                    help="eval/quickval.py の quick_eval を val-every 毎に呼ぶ"
                         "ケース数。0 で無効。best.pt の選択はこちらの lpips"
                         "(quick_eval, first 選択) が最小のときに行う (grain では"
                         "選ばない — 細い穴では grain が測れず盲点になる。"
                         "このファイル冒頭の docstring 参照)")
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

    model = StageFW2(base=base, k=a.k).to(a.device)
    if init_sd is not None:
        model.load_backbone(init_sd)
    n_par = sum(q.numel() for q in model.parameters())
    p(f"StageFW2 params: {n_par:,}  (base={base}, k={a.k}, crop={a.crop}, eps={a.eps})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    w0 = {name: v.detach().clone() for name, v in model.named_parameters()}

    perceptual = None
    if a.quickval_cases > 0:
        from eval.perceptual import Perceptual
        perceptual = Perceptual(device=a.device)

    fixed = None
    if a.overfit:
        b = next(it)
        img, mask = b["image"][:a.overfit].to(a.device), b["mask"][:a.overfit].to(a.device)
        fixed = random_crop(img, mask, a.crop)
        p(f"過学習テスト: 固定 {a.overfit} タイル (crop={a.crop})")

    hist = []
    best_lpips = 1e9         # quick_eval (first 選択) の lpips が最小のときのモデルを別に残す
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

            # **quick_eval で最終評価と同じ経路・同じ指標を測る (SPEC R11)。**
            # `validate()` の保留 grain は学習に使う混合マスク (電線などの
            # 細いものを含む) で測っており、細い穴では grain が盲目になる
            # (このファイル冒頭 docstring 参照)。best.pt の選択には使わない。
            qv_first = qv_oracle = None
            if a.quickval_cases > 0:
                model.eval()
                qv_first = quick_eval(make_query_fn(model, "first"),
                                      n_cases=a.quickval_cases, device=a.device,
                                      shard_dir=a.shards, perceptual=perceptual)
                # **oracle は診断専用。** GT を見て K 個の仮説から最良を選ぶ
                # 上限測定であり、`oracle - first` が多様性の指標になる
                # (このファイル冒頭 docstring、SPEC の「効かない条件」参照)。
                qv_oracle = quick_eval(make_query_fn(model, "oracle"),
                                       n_cases=a.quickval_cases, device=a.device,
                                       shard_dir=a.shards, perceptual=perceptual)
                model.train()
                if hist:
                    hist[-1]["quickval_first"] = qv_first
                    hist[-1]["quickval_oracle"] = qv_oracle
                d_psnr = qv_oracle["psnr"] - qv_first["psnr"]
                d_lpips = qv_oracle["lpips"] - qv_first["lpips"]
                p(f"  [quickval/first]  grain={qv_first['grain']:.4f}  "
                  f"psd={qv_first['psd']:.4f}  psnr={qv_first['psnr']:.2f}  "
                  f"lpips={qv_first['lpips']:.5f}")
                p(f"  [quickval/oracle] grain={qv_oracle['grain']:.4f}  "
                  f"psd={qv_oracle['psd']:.4f}  psnr={qv_oracle['psnr']:.2f}  "
                  f"lpips={qv_oracle['lpips']:.5f}")
                p(f"  [多様性] oracle-first psnr={d_psnr:+.2f} lpips={d_lpips:+.5f}  "
                  f"({'多様性あり (WTA が機能している兆候)' if d_psnr > 0.1 else '**多様性が薄い可能性 (WTA が機能していないかもしれない)**'})")

            mark = ""
            # 最良チェックポイントを別名で残す (StageA/StageF で last.pt 上書きにより
            # 最良モデルを失った教訓と同じ)。**判定は quick_eval (first 選択) の
            # lpips のみ** (grain では選ばない — SPEC R11)。
            if qv_first is not None and np.isfinite(qv_first["lpips"]) \
                    and qv_first["lpips"] < best_lpips:
                best_lpips = qv_first["lpips"]
                os.makedirs(a.out, exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                           "base": base, "k": a.k, "arch": "fw2",
                           "val_l1": vl, "val_grain": vg,
                           "quickval_first": qv_first, "quickval_oracle": qv_oracle},
                          os.path.join(a.out, "best.pt"))
                mark = "  <- best 更新 (quickval lpips が最小)"
            vwr_s = (np.array2string(vwr, precision=2, suppress_small=True)
                    if vwr is not None else "n/a")
            p(f"  {'':>6}  保留 L1 {vl:.5f}  保留 grain {vg:.4f} (理想 1.0)  "
              f"保留勝率 {vwr_s}{mark}")

        if step % 500 == 0 or step == a.steps:
            os.makedirs(a.out, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step,
                       "base": base, "k": a.k, "arch": "fw2"},
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
        ds = [(r["step"], r["quickval_oracle"]["psnr"] - r["quickval_first"]["psnr"],
              r["quickval_oracle"]["lpips"] - r["quickval_first"]["lpips"])
             for r in hist if r.get("quickval_first") is not None]
        if ds:
            p(f"  多様性 (oracle-first)  psnr {ds[0][1]:+.2f} -> {ds[-1][1]:+.2f}  "
              f"lpips {ds[0][2]:+.5f} -> {ds[-1][2]:+.5f}")
        p(f"  最良 (quickval lpips) = {best_lpips:.5f} (checkpoints/.../best.pt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
