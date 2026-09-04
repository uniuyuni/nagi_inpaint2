"""Stage F の追加学習 — 知覚損失 + 敵対的損失で「肉眼で LaMa に勝つ」を狙う
(train_stage_f.py を出発点にした続き学習器、SPEC §16)。

**背景 (なぜこのファイルが要るか)**

目標が「PSNR で勝つ」から「肉眼で LaMa に勝つ」に変わった (SPEC §16)。
`train_stage_f.py` (L1 + 勾配損失のみ) は平均値回帰でぼける: 実測で
grain 0.367 (LaMa 0.936、理想 1.0)、psd 0.2793 (LaMa 0.1026)。
**滑らかすぎるのが唯一かつ最大の敗因**であり、L1 を回し続けても解消しない
(実証済み)。したがって知覚損失と敵対的損失を追加し、`checkpoints/stage_f96/last.pt`
(base=96, 30k step 済み) から続きを学習する。

**評価指標と学習損失で別のネットワークを使う理由 (重要)**

`eval/perceptual.py` は VGG19 の特徴距離で評価する。学習の知覚損失に同じ
VGG19 を使うと、指標を直接最適化することになり「肉眼で良くなった」のか
「指標を騙せただけ」なのかが区別できなくなる。そこで学習側は **AlexNet**
(`~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` にキャッシュ済み)
を使う。新規ダウンロードはしない。

**損失の構成**

    loss = L1_hole + 0.3 * L1_known + 0.5 * grad
           + lam_perc * perceptual + lam_adv * adversarial

- L1_hole / L1_known / grad は `train_stage_f.py` の `step_loss` と同じ式
  (穴 L1 + 既知領域 L1 (重み0.3) + 勾配損失 (重み0.5))。
- perceptual: AlexNet の ReLU 直後 (層 1,4,7,9,11) の特徴をチャンネル方向 L2
  正規化してから差の二乗を取り、マスクをその層の解像度へ area 補間して
  加重平均する。マスク領域 (穴) だけを測る。`eval/perceptual.py` と同じ手順を
  AlexNet に対して行う。既定 lam-perc=1.0。
- adversarial: PatchGAN (70x70 相当) + hinge loss。生成器は `-D(fake).mean()`、
  識別器は `relu(1-D(real)).mean() + relu(1+D(fake)).mean()`。識別器の入力は
  画像3ch + マスク1ch (どこが穴かを識別器に教える)。BatchNorm は使わない
  (マスク入力で統計が壊れた過去の失敗があるため、StageF 本体と同じ理由で
  InstanceNorm を使う)。既定 lam-adv=0.1、`--lam-adv 0` で完全に無効化できる
  (この場合は識別器を一切呼ばない)。

過学習ゲート (`--overfit`) は StageA/StageF と同じ理由で必須:

  pixi run python train/train_stage_f2.py --init checkpoints/stage_f96/last.pt \
      --overfit 2 --steps 300 --crop 128 --device cpu   # 関門
  pixi run python train/train_stage_f2.py --init checkpoints/stage_f96/last.pt \
      --steps 8000                                       # 本学習 (mps)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nagi_inpaint2.stage_f import StageF                          # noqa: E402
from train.dataset import list_shards, make_loader, split_shards  # noqa: E402
from train.train_stage_f import grad_loss, random_crop            # noqa: E402


def p(*a):
    print(*a, flush=True)


# --- AlexNet 知覚損失 --------------------------------------------------------

# torchvision.models.alexnet().features を列挙して確認した ReLU 直後の位置。
# eval/perceptual.py の VGG19 版と同じ考え方 (複数深度の ReLU 出力を使う)。
_ALEXNET_RELU_LAYERS = (1, 4, 7, 9, 11)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class AlexNetPerceptual:
    """学習の知覚損失専用 (評価指標の VGG19 とは別ネットワーク)。

    `eval/perceptual.py` の `Perceptual` と同じ手順 (チャンネル L2 正規化 ->
    差の二乗 -> 層ごとの解像度へマスクを area 縮小して加重平均) を、勾配が
    `pred` へ流れる形で実装したもの。AlexNet の重みは
    `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` から読み、
    新規ダウンロードはしない。
    """

    def __init__(self, device: str = "cpu", layers=_ALEXNET_RELU_LAYERS):
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        model = torchvision.models.alexnet(weights=weights)
        self.features = model.features.to(device).eval()
        for param in self.features.parameters():
            param.requires_grad_(False)
        self.layers = tuple(layers)
        self._max_layer = max(self.layers)
        self.device = device
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self._mean = mean.to(device)
        self._std = std.to(device)

    def to(self, device: str) -> "AlexNetPerceptual":
        self.features = self.features.to(device)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.device = device
        return self

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """知覚空間 (値域 0..1.2 前後) を ImageNet 正規化済みに変換する。

        `eval/perceptual.py._prep` と同じ式 (`clamp(x/1.2, 0, 1)`)。
        """
        x = torch.clamp(x / 1.2, 0.0, 1.0)
        return (x - self._mean) / self._std

    def _extract(self, t: torch.Tensor) -> list[torch.Tensor]:
        want = set(self.layers)
        feats = []
        h = t
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in want:
                feats.append(h)
            if i >= self._max_layer:
                break
        return feats

    def __call__(self, pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """マスク領域 (穴) だけの知覚距離。`pred` への勾配は保つ。

        `pred` は生成器の出力なので勾配を流す必要がある (no_grad にしない)。
        `target` (GT) は勾配不要なので no_grad で計算しメモリを節約する。
        """
        fp = self._extract(self._prep(pred))
        with torch.no_grad():
            ft = self._extract(self._prep(target))

        layer_dists = []
        for xp, xt in zip(fp, ft):
            np_ = xp / (xp.norm(p=2, dim=1, keepdim=True) + 1e-10)
            nt = xt / (xt.norm(p=2, dim=1, keepdim=True) + 1e-10)
            d = (np_ - nt) ** 2
            d = d.mean(dim=1, keepdim=True)          # (B,1,h,w)
            # **mode="area" を使わない。** MPS の adaptive pool は入力が出力で
            # 割り切れる場合しか実装されておらず、AlexNet の特徴マップ
            # (63x63, 31x31 など) で必ず落ちる。bilinear は任意サイズで動き、
            # マスクを重みとして縮小する用途では area と実質同等。
            m = F.interpolate(mask, size=d.shape[-2:], mode="bilinear",
                              align_corners=False)
            val = (d * m).sum() / (m.sum() + 1e-8)
            layer_dists.append(val)
        return torch.stack(layer_dists).mean()


# --- PatchGAN 識別器 ----------------------------------------------------------

class PatchGAN(nn.Module):
    """70x70 相当の PatchGAN 識別器。SpectralNorm は使わず InstanceNorm を使う
    (StageF 本体と同じ理由: マスク入力で BatchNorm 統計が壊れた過去の失敗が
    あるため BatchNorm は使わない)。

    入力: 画像3ch + マスク1ch を連結した4ch (どこが穴かを識別器に教える)。
    出力: (N,1,h,w) のパッチごとロジット (hinge loss にそのまま渡す)。
    """

    def __init__(self, in_ch: int = 4, base: int = 64):
        super().__init__()
        # pix2pix の 70x70 PatchGAN と同じ構成 (最初の層だけ正規化を省く慣習)。
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 4, base * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img, mask], 1)
        return self.net(x)


def d_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """識別器の hinge loss: relu(1-D(real)).mean() + relu(1+D(fake)).mean()。"""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """生成器側の hinge loss: -D(fake).mean()。"""
    return -d_fake.mean()


# --- 生成器の損失 (再構成 + 知覚 + 敵対的) -------------------------------------

def step_loss_gen(model, disc, img, mask, *, lam_grad: float, lam_known: float,
                   lam_perc: float, lam_adv: float, perceptual: AlexNetPerceptual,
                   lam_hole: float = 1.0):
    """生成器側の合計損失と、内訳を個別に返す。

    `disc` が None または `lam_adv <= 0` のときは識別器を一切呼ばない
    (adv=0 tensor を返すだけ)。「--lam-adv 0 で完全に無効化できる」という
    仕様を関数レベルで保証するための分岐。

    戻り値: (total_loss, {"l1": ..., "l1k": ..., "grad": ..., "perc": ..., "adv": ...})
    total_loss が None の場合は穴が空でスキップすべきことを示す (StageA/StageF と同じ規約)。
    """
    pred = model(img, mask)
    m = mask.expand_as(pred)
    if m.sum() < 1:
        return None, None, None

    l1 = ((pred - img).abs() * m).sum() / (m.sum() + 1e-6)
    k = 1.0 - m
    l1k = ((pred - img).abs() * k).sum() / (k.sum() + 1e-6)
    lg = grad_loss(pred * mask, img * mask)
    perc = perceptual(pred, img, mask)

    # **穴の中の L1 は既定で重み 0 にできるようにする。**
    # L1 は平均値を最適解とするので、穴の中に L1 を課すと**ぼけを強制する**。
    # 実測: lam_hole=1.0 / lam_adv=0.1 で 3000 step 回しても grain は
    # 0.322 -> 0.328 と動かず、知覚距離は 0.0059 -> 0.0061 と悪化した。
    # LaMa は穴の L1 を 0、既知領域の L1 を 10、敵対 10、知覚 30 にしている。
    # 穴の中は知覚損失と敵対的損失だけで決めるのが正しい。
    total = lam_hole * l1 + lam_known * l1k + lam_grad * lg + lam_perc * perc

    if disc is not None and lam_adv > 0:
        d_fake = disc(pred, mask)
        adv = g_hinge_loss(d_fake)
        total = total + lam_adv * adv
    else:
        adv = torch.zeros((), device=img.device)

    parts = {"l1": float(l1.detach()), "l1k": float(l1k.detach()),
             "grad": float(lg.detach()), "perc": float(perc.detach()),
             "adv": float(adv.detach())}
    return total, pred, parts


def step_loss_disc(disc, pred: torch.Tensor, img: torch.Tensor, mask: torch.Tensor):
    """識別器側の hinge loss。`pred` は呼び出し側で detach 済みを渡すこと
    (生成器パラメータへ勾配を流さないため)。
    """
    d_real = disc(img, mask)
    d_fake = disc(pred, mask)
    return d_hinge_loss(d_real, d_fake)


# --- 保留評価: L1 と知覚損失 ---------------------------------------------------

@torch.no_grad()
def validate(model, perceptual: AlexNetPerceptual, it, n: int, device: str, crop: int):
    """保留 shard での L1 と知覚損失 (マスク領域)。両方を個別に返す。"""
    model.eval()
    tot_l1 = tot_perc = 0.0
    c = 0
    for _ in range(n):
        b = next(it)
        img, mask = b["image"].to(device), b["mask"].to(device)
        img, mask = random_crop(img, mask, crop)
        m = mask.expand(-1, 3, -1, -1)
        if m.sum() < 1:
            continue
        pred = model(img, mask)
        tot_l1 += float(((pred - img).abs() * m).sum() / m.sum())
        tot_perc += float(perceptual(pred, img, mask))
        c += 1
    model.train()
    if c == 0:
        return float("nan"), float("nan")
    return tot_l1 / c, tot_perc / c


def load_generator(init_path: str | None, base_default: int, n_ffc: int, device: str):
    """`--init` があれば checkpoint から base を復元してロードする。

    **既定値 (base_default) で StageF を作ってから load_state_dict すると、
    checkpoint の base (例: 96) と形が合わずロードに失敗する** (過去に実際に
    失敗した)。checkpoint の `out.weight` の形 (3, base, 3, 3) から base を
    復元してからモデルを構築する。
    """
    if init_path is None:
        p(f"--init 未指定: base={base_default} で新規初期化する "
          f"(train_stage_f2.py は続き学習用途を想定しているため通常は --init を渡すこと)")
        model = StageF(base=base_default, n_ffc=n_ffc).to(device)
        return model, None

    ckpt = torch.load(init_path, map_location=device)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    base = int(sd["out.weight"].shape[1])
    model = StageF(base=base, n_ffc=n_ffc).to(device)
    model.load_state_dict(sd, strict=True)
    p(f"checkpoint 復元: {init_path}  base={base}  (元 step={ckpt.get('step', '?')})")
    return model, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--init", default=None,
                    help="既存 StageF checkpoint (例: checkpoints/stage_f96/last.pt)。"
                         "base は checkpoint から自動復元する。")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256,
                    help="全解像度 (512) がメモリに乗らない場合のランダムクロップ辺")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="続き学習なので元の 2e-4 より小さくする")
    ap.add_argument("--lr-d", type=float, default=1e-4, help="識別器の学習率")
    ap.add_argument("--lam-grad", type=float, default=0.5)
    ap.add_argument("--lam-known", type=float, default=0.3)
    ap.add_argument("--lam-perc", type=float, default=1.0)
    ap.add_argument("--lam-adv", type=float, default=0.1,
                    help="0 にすると敵対的損失を完全に無効化する (識別器を呼ばない)")
    ap.add_argument("--lam-hole", type=float, default=1.0,
                    help="穴の中の L1 の重み。知覚品質を狙うときは 0 にする")
    ap.add_argument("--base", type=int, default=32,
                    help="--init 未指定時のみ使う既定 base")
    ap.add_argument("--n-ffc", type=int, default=4)
    ap.add_argument("--mask-kind", default=None)
    ap.add_argument("--out", default="checkpoints/stage_f2")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=12)
    ap.add_argument("--ckpt-every", type=int, default=500)
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

    model, base = load_generator(a.init, a.base, a.n_ffc, a.device)
    n_par = sum(q.numel() for q in model.parameters())
    p(f"StageF params: {n_par:,}  base={base or a.base}  (全解像度で動作、crop={a.crop})")

    perceptual = AlexNetPerceptual(device=a.device)

    disc = None
    opt_d = None
    if a.lam_adv > 0:
        disc = PatchGAN().to(a.device)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=a.lr_d)
        n_par_d = sum(q.numel() for q in disc.parameters())
        p(f"PatchGAN params: {n_par_d:,}  lam_adv={a.lam_adv}")
    else:
        p("lam_adv=0: 識別器は構築しない (敵対的損失は完全に無効)")

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
    best_perc = 1e9        # 保留の知覚損失が最良のときのモデルを別に残す
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        if fixed is not None:
            img, mask = fixed
        else:
            b = next(it)
            img, mask = b["image"].to(a.device), b["mask"].to(a.device)
            img, mask = random_crop(img, mask, a.crop)

        total, pred, parts = step_loss_gen(
            model, disc, img, mask, lam_grad=a.lam_grad, lam_known=a.lam_known,
            lam_hole=a.lam_hole,
            lam_perc=a.lam_perc, lam_adv=a.lam_adv, perceptual=perceptual)
        if total is None:
            continue

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        d_loss_val = float("nan")
        if disc is not None and a.lam_adv > 0:
            d_loss = step_loss_disc(disc, pred.detach(), img, mask)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()
            d_loss_val = float(d_loss.detach())

        if step % a.log_every == 0:
            with torch.no_grad():
                # buffers を含めない。named_parameters() のみで |Δw|max を計算する
                # (StageA/StageF で BatchNorm running_var 暴走を誤読した教訓)。
                dw = max(float((q - w0[k]).abs().max())
                         for k, q in model.named_parameters())
            el = time.perf_counter() - t0
            rec = dict(step=step, total=float(total.detach()), **parts,
                       d_loss=d_loss_val, dw=dw, lr=float(sched.get_last_lr()[0]),
                       sec_per_step=el / step)
            hist.append(rec)
            p(f"  {step:>6}/{a.steps}  l1 {rec['l1']:.5f}  l1k {rec['l1k']:.5f}  "
              f"grad {rec['grad']:.5f}  perc {rec['perc']:.5f}  adv {rec['adv']:.5f}  "
              f"d_loss {d_loss_val:.5f}  |Δw|max {dw:.3e}  {rec['sec_per_step']*1000:.0f}ms/step")

        if step % a.val_every == 0 or step == a.steps:
            vl, vperc = validate(model, perceptual, vit, a.val_batches, a.device, a.crop)
            if hist:
                hist[-1]["val_l1"] = vl
                hist[-1]["val_perc"] = vperc
            mark = ""
            # 最良チェックポイントを別名で残す (StageA/StageF で last.pt 上書きにより
            # 最良モデルを失った教訓と同じ)。基準は保留の知覚損失 (小さいほど良い)。
            if np.isfinite(vperc) and vperc < best_perc:
                best_perc = vperc
                os.makedirs(a.out, exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step, "base": base or a.base,
                            "val_l1": vl, "val_perc": vperc, **parts},
                           os.path.join(a.out, "best.pt"))
                mark = "  <- best 更新"
            p(f"  {'':>6}  保留 L1 {vl:.5f}  保留 知覚損失 {vperc:.5f}{mark}")

        if step % a.ckpt_every == 0 or step == a.steps:
            os.makedirs(a.out, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step, "base": base or a.base,
                        **parts}, os.path.join(a.out, "last.pt"))
            with open(os.path.join(a.out, "history.jsonl"), "w") as f:
                for r in hist:
                    f.write(json.dumps(r) + "\n")

    if len(hist) >= 2:
        p(f"\n=== 学習前後 ===")
        p(f"  l1       {hist[0]['l1']:.5f} -> {hist[-1]['l1']:.5f}")
        p(f"  grad     {hist[0]['grad']:.5f} -> {hist[-1]['grad']:.5f}")
        p(f"  perc     {hist[0]['perc']:.5f} -> {hist[-1]['perc']:.5f}")
        p(f"  adv      {hist[0]['adv']:.5f} -> {hist[-1]['adv']:.5f}")
        p(f"  |Δw|max  {hist[-1]['dw']:.3e} "
          f"({'パラメータが動いている' if hist[-1]['dw'] > 1e-5 else '**動いていない**'})")
        p(f"  最良      保留知覚損失 {best_perc:.5f} (checkpoints/.../best.pt)")
        vs = [(r['step'], r['val_l1'], r['val_perc'])
              for r in hist if r.get('val_perc') is not None]
        if len(vs) >= 2:
            p(f"  保留 L1  {vs[0][1]:.5f} -> {vs[-1][1]:.5f}")
            p(f"  保留 知覚損失  {vs[0][2]:.5f} -> {vs[-1][2]:.5f} "
              f"({'改善' if vs[-1][2] < vs[0][2] else '**悪化/横ばい**'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
