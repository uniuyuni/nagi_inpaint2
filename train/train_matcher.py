"""照合コストの学習 (SPEC §4.2)。

**何を学ぶのか**

    神託 (正解を見てブロックごとに最良オフセットを選ぶ)   33.27 dB
    実測 (周囲の既知画素の照合コストで選ぶ)              23.88 dB
    LaMa                                            29.53 dB

差の 9.4 dB は「正解なしでどのオフセットが良いかを当てる精度」。
FeatureNet の仕事はこれだけ。**探索そのものは学習対象ではない。**

**前回38時間を失った反省として組み込んだこと**

  1. 微分不可能な argmin を勾配経路に含めない。
     コストの順位を実誤差の順位に合わせるだけなので、探索の外側で学習できる
  2. 候補と実誤差を事前計算 (data/cache_candidates.py)。1 step が2桁速い
  3. **パラメータの変化量を毎回記録する。**
     前回はゼロ初期化で勾配が厳密に 0 だったのに、loss と entropy が
     τ の焼き鈍しで動いていたため 20,000 step 気づかなかった
  4. 主指標を regret にする。損失ではなく「選択の悪さ」を直接見る

     regret = 選んだ候補の実誤差 / 最良候補の実誤差
     1.0 なら神託と同じ。現状の手作り特徴がいくつかを測り、それを下回るかを見る

  pixi run python train/train_matcher.py --steps 500
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

from nagi_inpaint2.stage_b import FeatureNet                     # noqa: E402
from nagi_inpaint2.stage_b_search import _box, _shift            # noqa: E402
from nagi_inpaint2.stage_b_simple import _features               # noqa: E402

CACHE = "cache/match_train.npz"
MIN_COV, MIN_VALID = 0.25, 0.9
PATCH = 7


def p(*a):
    print(*a, flush=True)


class Cache:
    def __init__(self, path: str, device: str):
        d = np.load(path)
        self.imgs = d["imgs"]
        self.masks = d["masks"]
        self.off_idx = d["off_idx"]
        self.errs = d["errs"]
        self.covs = d["covs"]
        self.vfracs = d["vfracs"]
        self.offsets = d["offsets"]
        self.block = int(d["block"])
        self.scale = float(d["img_scale"])
        self.device = device
        self.n = len(self.imgs)
        self.tile = self.imgs.shape[1]

    def get(self, i: int):
        img = torch.from_numpy(
            self.imgs[i].astype(np.float32) / self.scale
        ).permute(2, 0, 1)[None].to(self.device)
        m = np.unpackbits(self.masks[i])[:self.tile ** 2].reshape(
            self.tile, self.tile).astype(np.float32)
        mask = torch.from_numpy(m[None, None]).to(self.device)
        err = torch.from_numpy(self.errs[i]).to(self.device)       # (N,h,w)
        cov = torch.from_numpy(self.covs[i]).to(self.device)       # (h,w)
        vf = torch.from_numpy(self.vfracs[i]).to(self.device)      # (N,h,w)
        offs = [tuple(self.offsets[j]) for j in self.off_idx[i]]
        return img, mask, err, cov, vf, offs


def block_costs(img, mask, offs, feat, block: int, patch: int = PATCH,
                cov=None):
    """キャッシュされた候補オフセットについてのみコストを出す。

    全探索 (2400候補) ではなく 64候補なので 2桁速い。
    """
    src = 1.0 - mask
    f = (_features(img) if feat is None else feat(img, mask)) * src
    r = patch // 2
    out, val = [], []
    for oy, ox in offs:
        vs = _shift(src, int(oy), int(ox))
        w = src * vs
        wb = _box(w, r)
        c = _box(((f - _shift(f, int(oy), int(ox))) ** 2).sum(1, keepdim=True) * w,
                 r) / (wb + 1e-6)
        out.append(F.avg_pool2d(c * mask, block))
        # **マスク被覆で正規化する。** ブロック全体に対する割合で測ると、
        # 被覆が 50% 未満のブロックが全て弾かれ、対象がゼロになる。
        val.append(F.avg_pool2d((wb > 0.05).float() * mask, block))
    C = torch.cat(out, 1)[0]
    V = torch.cat(val, 1)[0]
    if cov is not None:
        V = V / (cov[None] + 1e-8)
    return C, V                                              # (N,h,w)


def regret(cost, err, usable):
    """選んだ候補の実誤差 / 最良候補の実誤差。1.0 が神託。"""
    big = torch.full_like(cost, float("inf"))
    c = torch.where(usable, cost, big)
    e = torch.where(usable, err, big)
    pick = c.argmin(0)
    got = err.gather(0, pick[None])[0]
    best = e.min(0).values
    ok = torch.isfinite(best) & (best > 1e-9) & usable.any(0)
    if not ok.any():
        return float("nan")
    return float((got[ok] / best[ok]).median())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default="checkpoints/matcher")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--overfit", type=int, default=0,
                    help=">0 なら先頭 N タイルだけで過学習テスト")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    cache = Cache(a.cache, a.device)
    n_use = a.overfit if a.overfit else cache.n
    p(f"キャッシュ {cache.n} タイル / 使用 {n_use} / ブロック {cache.block}px")

    net = FeatureNet().to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    w0 = {k: v.detach().clone() for k, v in net.state_dict().items()}
    p(f"FeatureNet params: {sum(q.numel() for q in net.parameters()):,}")

    # 起点: 手作り特徴のみ (学習なし) の regret
    base = []
    for i in range(min(n_use, 16)):
        img, mask, err, cov, vf, offs = cache.get(i)
        with torch.no_grad():
            c, v = block_costs(img, mask, offs, None, cache.block, cov=cov)
        # v (パッチ文脈の可用性) は vf (参照元の有効性) と別の量で、
        # 同時に成立しない。キャッシュ済みの vf が必要な条件を捉えているので使わない。
        usable = (vf > MIN_VALID) & (cov > MIN_COV)[None]
        r = regret(c, err, usable)
        if np.isfinite(r):
            base.append(r)
    p(f"起点 (手作り特徴のみ) regret 中央値 {np.median(base):.2f}  (1.0 が神託)\n")

    rng = np.random.default_rng(0)
    hist = []
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        i = int(rng.integers(n_use))
        img, mask, err, cov, vf, offs = cache.get(i)
        cost, v = block_costs(img, mask, offs, net, cache.block, cov=cov)
        usable = (vf > MIN_VALID) & (cov > MIN_COV)[None]

        # 使える候補が2つ以上あるブロックだけを教師にする
        sel = usable.sum(0) >= 2
        if not sel.any():
            continue
        big = torch.tensor(1e6, device=a.device)
        C = torch.where(usable, cost, big)[:, sel]           # (N, B)
        E = torch.where(usable, err, big)[:, sel]
        # 実誤差の順位に合わせる (listwise)。低誤差の候補へ確率を寄せる
        tgt = torch.softmax(-E / (E.min(0, keepdim=True).values + 1e-6) / a.tau, 0)
        logp = torch.log_softmax(-C / (C.min(0, keepdim=True).values + 1e-6) / a.tau, 0)
        loss = -(tgt * logp).sum(0).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0:
            with torch.no_grad():
                r = regret(cost.detach(), err, usable)
                dw = max(float((net.state_dict()[k] - w0[k]).abs().max())
                         for k in w0 if w0[k].dtype.is_floating_point)
            el = time.perf_counter() - t0
            rec = dict(step=step, loss=float(loss.detach()), regret=r,
                       dw=dw, sec_per_step=el / step)
            hist.append(rec)
            p(f"  {step:>5}/{a.steps}  loss {rec['loss']:.5f}  "
              f"regret {r:6.2f}  |Δw|max {dw:.3e}  {rec['sec_per_step']*1000:.0f}ms/step")

    os.makedirs(a.out, exist_ok=True)
    torch.save({"model": net.state_dict(), "step": a.steps},
               os.path.join(a.out, "last.pt"))
    with open(os.path.join(a.out, "history.jsonl"), "w") as f:
        for r in hist:
            f.write(json.dumps(r) + "\n")

    if len(hist) >= 2:
        p(f"\n=== 学習前後 ===")
        p(f"  loss    {hist[0]['loss']:.5f} -> {hist[-1]['loss']:.5f}")
        p(f"  regret  {hist[0]['regret']:.2f} -> {hist[-1]['regret']:.2f} "
          f"(起点 {np.median(base):.2f}, 神託 1.00)")
        p(f"  |Δw|max {hist[-1]['dw']:.3e} "
          f"({'パラメータが動いている' if hist[-1]['dw'] > 1e-5 else '**動いていない**'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
