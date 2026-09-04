"""Phase 0b — 2 arm 対照実験: 目標を替えるだけで鮮鋭さが保たれるか。

**仮説** (docs/DESIGN_V2_JA.md §1.2c)

  目標が「正解画像」→ 穴の中身は入力から復元不能 → 条件付き分布が広い
                     → L1 の最適解は平均 → **ぼける**
  目標が「決定的な教師の出力」→ 条件付き分布はデルタ関数
                     → L1 の最適解はその値 → **ぼけない**

**対照実験の設計**

同じ npz (`train/distill_gen.py` が作る) から、**同じ入力・同じマスク・
同じ初期重み・同じ手順**で 2 つの arm を回し、**目標だけを替える**。

    --target gt    : 目標 = 正解画像          (v1 の設定。既知の結果 grain≈0.18)
    --target lama  : 目標 = LaMa の出力       (決定的教師)

差が出れば主張が支持され、出なければ反証される。

**損失の組み方 (重要)**

マスクは画像の平均 9.3% しかない。画像全体で `abs().mean()` を取ると
**穴への勾配が 9% まで薄まり、残り 91% は「見えている画素のコピー」に費やされる**。
初版でこれをやり、200 件の暗記すら進まなかった (d_target が 800 step 平坦)。
穴の中と外を**別々に正規化**し、外側に `--lam-known` を掛ける
(v1 の `train_stage_f.step_loss` と同じ組み方)。

**効かない条件 (R15: 実装前に書く)**

  - 生徒の容量不足 → **両 arm ともぼける**。この場合 arm 間に差は出ない
  - 教師 T が入力に対して非滑らか → arm lama もぼける (= 主張の反証)
  - 学習不足 → 両方ぼける。だから**同一 step 数で比較する**

**主要指標は grain ではない**

初稿では grain_ratio を主指標にしかけたが、**preflight で乱数初期化の
モデルが grain 0.951 を出した**。grain は「正しいテクスチャ」と「ただの雑音」を
区別できない。v1 の Stage G が `grain 0.934 だが中身は青黒い粒々` で
踏んだ罠そのものである。

  d_target  : **目標との知覚距離** (マスク領域、VGG19 特徴)。← 合否はこれで決める
  l1_target : 目標との L1 距離 (穴の中)
  grain     : 診断値として並記するだけ
  psnr_gt   : 参考値

**陰性対照 (R12)** を学習前に 1 度測り、基準線を置く:

  teacher_self : 目標そのもの      → d_target の下限
  blur         : 目標をぼかしたもの → 「ぼけ」がどの値か
  copy_outside : 穴を平均で埋める   → 最も無情報な出力

**学習後の d_target が blur 基準より小さくなって初めて「ぼけていない」と言える。**

  pixi run python train/distill_train.py --target lama --steps 6000
  pixi run python train/distill_train.py --target gt   --steps 6000
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

from eval.metrics import grain_ratio, psnr                       # noqa: E402
from eval.perceptual import Perceptual                            # noqa: E402
from nagi_inpaint2.stage_f import StageF                          # noqa: E402


def p(*a):
    print(*a, flush=True)


class Pack:
    """`distill_gen.py` が作った npz を RAM に載せて、バッチを切り出す。"""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.img = z["img"]            # (N,3,C,C) float16
        self.lama = z["lama"]
        self.mask = z["mask"]          # (N,1,C,C) uint8
        self.meta = json.loads(str(z["meta"]))
        self.crop = int(z["crop"])
        self.n = self.img.shape[0]

    def batch(self, idx, device: str):
        i = torch.from_numpy(self.img[idx].astype(np.float32)).to(device)
        l = torch.from_numpy(self.lama[idx].astype(np.float32)).to(device)
        m = torch.from_numpy(self.mask[idx].astype(np.float32)).to(device)
        return i, l, m


def _blur(x: np.ndarray, k: int = 5) -> np.ndarray:
    """箱平均でぼかす。**指標の陰性対照用** (R12)。"""
    import scipy.ndimage as ndi
    return np.stack([ndi.uniform_filter(x[..., c], k) for c in range(x.shape[-1])], -1)


def _metrics(comp: np.ndarray, img: np.ndarray, tgt_c: np.ndarray,
             msk: np.ndarray, perc) -> dict:
    """1 ケース分。**主要指標は「目標との知覚距離」であって grain ではない。**

    grain は「正しいテクスチャ」と「ただの雑音」を区別できない
    (乱数初期化のモデルでも grain≈0.95 が出る。v1 の Stage G が
     `grain 0.934 だが中身はノイズ` で踏んだ罠)。
    したがって合否は **d_target (目標との知覚距離)** で判断し、
    grain は補助的な診断値として並記するだけにする。
    """
    return dict(
        d_target=perc(comp, tgt_c, msk),
        l1_target=float(np.abs(comp[msk] - tgt_c[msk]).mean()),
        grain=grain_ratio(comp, msk),
        psnr_gt=psnr(comp, img, msk),
    )


def evaluate(model, pack: "Pack", device: str, n: int, target: str,
             perc) -> dict:
    """保留データで測る。**合成は運用と同じ (マスク外は原画素)。**"""
    model.eval()
    acc: dict[str, list] = {}
    with torch.inference_mode():
        for k in range(min(n, pack.n)):
            i, l, m = pack.batch(slice(k, k + 1), device)
            o = model(i, m)
            comp = torch.where(m > 0.5, o, i)
            c = comp[0].permute(1, 2, 0).float().cpu().numpy()
            img = i[0].permute(1, 2, 0).float().cpu().numpy()
            lam = l[0].permute(1, 2, 0).float().cpu().numpy()
            msk = m[0, 0].cpu().numpy() > 0.5
            lam_c = np.where(msk[..., None], lam, img)
            tgt_c = img if target == "gt" else lam_c

            r = _metrics(c, img, tgt_c, msk, perc)
            r["grain_target"] = grain_ratio(tgt_c, msk)
            for kk, vv in r.items():
                if np.isfinite(vv):
                    acc.setdefault(kk, []).append(vv)
    model.train()
    med = lambda v: float(np.median(v)) if v else float("nan")   # noqa: E731
    return {kk: med(vv) for kk, vv in acc.items()}


def controls(pack: "Pack", device: str, n: int, target: str, perc) -> dict:
    """指標の陰性対照 (R12)。学習の前に 1 度だけ測って基準線を置く。

      teacher_self : 目標そのもの (下限。0 に近いはず)
      blur         : 目標をぼかしたもの (「ぼけ」がどの値になるか)
      copy_outside : 穴を周囲の平均で埋めたもの (最も無情報な出力)

    学習後の d_target がこれらのどこに落ちるかで、
    「教師に追随した」のか「ぼけただけ」なのかを判定できる。
    """
    rows: dict[str, list] = {}
    for k in range(min(n, pack.n)):
        i, l, m = pack.batch(slice(k, k + 1), device)
        img = i[0].permute(1, 2, 0).float().cpu().numpy()
        lam = l[0].permute(1, 2, 0).float().cpu().numpy()
        msk = m[0, 0].cpu().numpy() > 0.5
        lam_c = np.where(msk[..., None], lam, img)
        tgt_c = img if target == "gt" else lam_c

        flat = img.copy()
        flat[msk] = img[~msk].mean(0)
        cands = dict(teacher_self=tgt_c, blur=np.where(msk[..., None],
                                                       _blur(tgt_c, 7), img),
                     copy_outside=flat)
        for nm, cand in cands.items():
            r = _metrics(cand, img, tgt_c, msk, perc)
            for kk, vv in r.items():
                if np.isfinite(vv):
                    rows.setdefault(f"{nm}.{kk}", []).append(vv)
    return {kk: float(np.median(vv)) for kk, vv in rows.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("gt", "lama"), required=True,
                    help="gt = 正解画像 (v1 の設定) / lama = 決定的教師の出力")
    ap.add_argument("--train", default="cache/distill_train.npz")
    ap.add_argument("--val", default="cache/distill_val.npz")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam-known", type=float, default=0.3,
                    help="既知領域の L1 の重み。穴の中は常に 1.0 で正規化する")
    ap.add_argument("--seed", type=int, default=0,
                    help="両 arm で同じ値にすること。初期重みを揃えるため")
    ap.add_argument("--overfit", type=int, default=0,
                    help=">0 ならこの件数だけを繰り返し、暗記能力を測る")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-cases", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    out = a.out or f"checkpoints/distill_{a.target}" + ("_of" if a.overfit else "")
    os.makedirs(out, exist_ok=True)

    tr = Pack(a.train)
    if a.overfit:
        tr.n = min(a.overfit, tr.n)
    va = Pack(a.val) if os.path.exists(a.val) and not a.overfit else tr
    p(f"arm={a.target}  train {tr.n} 件 / val {va.n} 件  crop {tr.crop}")

    # **両 arm で初期重みを揃える** — 対照実験の条件
    torch.manual_seed(a.seed)
    model = StageF(base=a.base).to(a.device)
    npar = sum(q.numel() for q in model.parameters())
    p(f"StageF base={a.base}  {npar/1e6:.2f}M params  seed={a.seed}")

    perc = Perceptual(device=a.device)
    ctrl = controls(va, a.device, a.val_cases, a.target, perc)
    p("\n=== 指標の陰性対照 (R12) — 学習前に基準線を置く ===")
    for nm in ("teacher_self", "blur", "copy_outside"):
        p(f"  {nm:14s} d_target {ctrl.get(nm+'.d_target', float('nan')):.5f}"
          f"  l1 {ctrl.get(nm+'.l1_target', float('nan')):.5f}"
          f"  grain {ctrl.get(nm+'.grain', float('nan')):.3f}")
    p("  -> 学習後の d_target が blur より小さければ「ぼけていない」と言える\n")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    rng = np.random.default_rng(a.seed + 100)

    hist = open(os.path.join(out, "history.jsonl"), "a")
    hist.write(json.dumps(dict(step=0, controls=ctrl)) + "\n")
    hist.flush()
    t0 = time.perf_counter()
    run = 0.0

    for step in range(1, a.steps + 1):
        idx = rng.integers(0, tr.n, size=a.batch)
        i, l, m = tr.batch(idx, a.device)
        tgt = i if a.target == "gt" else l

        o = model(i, m)
        # **穴の中と外を別々に正規化する。** 画像全体の平均を取ると、
        # マスクは平均 9.3% しかないので穴への勾配が 9% まで薄まる
        # (初版でこれをやり、200件の暗記すら進まなかった)。
        # v1 の train_stage_f.step_loss と同じ組み方 (lam_known=0.3)。
        mm = m.expand_as(o)
        d = (o - tgt).abs()
        l1_hole = (d * mm).sum() / (mm.sum() + 1e-6)
        l1_known = (d * (1 - mm)).sum() / ((1 - mm).sum() + 1e-6)
        loss = l1_hole + a.lam_known * l1_known
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += float(loss.detach())

        if step % a.log_every == 0:
            el = time.perf_counter() - t0
            row = dict(step=step, loss=run / a.log_every,
                       lr=sched.get_last_lr()[0], sec_per_step=el / step)
            run = 0.0
            p(f"  step {step:5d}  loss {row['loss']:.5f}  "
              f"{row['sec_per_step']:.2f} s/step  "
              f"残り約 {(a.steps-step)*row['sec_per_step']/60:.0f} 分")
            hist.write(json.dumps(row) + "\n")
            hist.flush()

        if step % a.val_every == 0 or step == a.steps:
            v = evaluate(model, va, a.device, a.val_cases, a.target, perc)
            vs_blur = v["d_target"] / max(ctrl.get("blur.d_target", 1e-9), 1e-9)
            p(f"  [val] step {step}  d_target {v['d_target']:.5f} "
              f"(ぼけ基準比 {vs_blur:.2f})  l1 {v['l1_target']:.5f}  "
              f"grain {v['grain']:.3f} (目標 {v['grain_target']:.3f})  "
              f"psnr {v['psnr_gt']:.2f}")
            hist.write(json.dumps(dict(step=step, **v)) + "\n")
            hist.flush()
            torch.save(dict(model=model.state_dict(), step=step, base=a.base,
                            target=a.target, val=v, controls=ctrl),
                       os.path.join(out, "last.pt"))
        if a.device == "mps" and step % 200 == 0:
            torch.mps.empty_cache()

    p(f"\n完了 {(time.perf_counter()-t0)/60:.1f} 分 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
