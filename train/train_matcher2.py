"""照合コスト学習 v2 (SPEC §4.2)。

`train/train_matcher.py` (v1) との違いは2つ。

**1. query 上で学習する。** 推論 (`stage_b_search.solve_blocks(..., query=q)`) は
Stage A の低周波補完で埋めた `query` を照合対象にする。v1 は `img` (GT そのもの)
を照合対象に学習していたため、推論時の入力分布 (ぼけた低周波+実画素) とずれていた。
v2 は `data/cache_candidates2.py` が保存した `query` を常に使い、
学習・推論とも呼び出しを `feat(query_img, zeros_mask)` に統一する
(`solve_blocks` の query 分岐 `feat(qimg, torch.zeros_like(mask))` と同じ)。

**2. 損失を pairwise hinge にする。** v1 は softmax の listwise KL だったが、
全候補 (BIG を含む) を同じ温度で揃えるため、無効候補の混入や誤差スケールの
タイル間ばらつきに損失が敏感だった。ペアごとに「どちらが良いか」だけを
学ぶ hinge のほうが、誤差の絶対値ではなく順位だけを教師にするという設計意図
(SPEC の「探索そのものは学習対象ではない」) に忠実。

前回38時間を失った反省 (v1 のコメント参照) は引き続き有効:

  1. 微分不可能な argmin を勾配経路に含めない。ここでは argmin は一切登場せず、
     ペアの相対コストだけを損失にする。
  2. 候補と実誤差は事前計算済み (data/cache_candidates2.py)。
  3. パラメータの変化量 `|Δw|max` は **named_parameters() のみ** で測る
     (buffers を含めると BatchNorm 統計の暴走を学習量と誤読する — stage_a の教訓)。
     FeatureNet は BatchNorm を持たないが、慣習として同じガードを踏襲する。
  4. 主指標は regret (選んだ候補の実誤差 / 最良候補の実誤差)。
     **学習に使っていない cache 内 val タイル (末尾10%)** で測る。

  pixi run python train/train_matcher2.py --steps 2000
  pixi run python train/train_matcher2.py --overfit 4 --steps 300   # 関門
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

from nagi_inpaint2.stage_b import FeatureNet
from nagi_inpaint2.stage_b_search import FEAT_RMS                          # noqa: E402
from nagi_inpaint2.stage_b_search import BLOCK, MIN_W, PATCH, _shift  # noqa: E402
from nagi_inpaint2.stage_b_simple import _box                         # noqa: E402

CACHE = "cache/match_train2.npz"
MIN_COV, MIN_VALID = 0.25, 0.9
BIG = 1e6                # solve_blocks / cache_candidates2 と同じ「無効オフセット」の値
ORDER_RATIO = 0.7         # err_i < ORDER_RATIO * err_j で「i が明確に良い」とみなす
HINGE_MARGIN = 0.1
MAX_PAIRS = 256


def p(*a):
    print(*a, flush=True)


# --- コスト計算 (solve_blocks の query 分岐と同一式) ------------------------

def block_costs(query: torch.Tensor, mask: torch.Tensor, offs, feat,
                block: int = BLOCK, patch: int = PATCH) -> torch.Tensor:
    """query 画像上で、キャッシュされたオフェセットのみについてブロックコストを返す。

    `stage_b_search.solve_blocks(img, mask, feat, query=query)` の query 分岐と
    **厳密に同一の式**にしてある (query パスでは qval=ones、w=vs)。
    全オフセット (2400候補) を評価する solve_blocks と違い、ここでは
    `offs` に渡された少数 (通常64) の候補だけを評価するので学習が2桁速い。
    テスト (tests/test_matcher2.py) で solve_blocks の対応列と一致することを確認している。

    戻り値: (len(offs), h, w)
    """
    src = 1.0 - mask
    qval = torch.ones_like(mask)
    f = feat(query, torch.zeros_like(mask)) * qval
    # 推論側 (solve_blocks) と同じ特徴スケール正規化。順位は定数倍で不変なので
    # 学習の損失値は変わらないが、両者の式を厳密に一致させておく (テストで検証)。
    f = f * (FEAT_RMS / (f.pow(2).mean().sqrt() + 1e-12))
    r = patch // 2
    out = []
    for oy, ox in offs:
        oy, ox = int(oy), int(ox)
        vs = _shift(src, oy, ox)
        w = qval * vs
        wb = _box(w, r)
        c = _box(((f - _shift(f, oy, ox)) ** 2).sum(1, keepdim=True) * w, r) / (wb + 1e-6)
        bad = (vs <= 0.5) | (wb < MIN_W)
        c = torch.where(bad, torch.full_like(c, BIG), c)
        out.append(F.avg_pool2d(c * mask, block))
    return torch.cat(out, 1)[0]                                # (N,h,w)


# --- 損失: pairwise hinge ---------------------------------------------------

def pairwise_hinge_loss(cost: torch.Tensor, err: torch.Tensor, usable: torch.Tensor,
                        cov: torch.Tensor, rng: np.random.Generator,
                        min_cov: float = MIN_COV, order_ratio: float = ORDER_RATIO,
                        margin: float = HINGE_MARGIN, max_pairs: int = MAX_PAIRS):
    """コストの順位を実誤差の順位に合わせる pairwise hinge 損失。

    **argmin を経由しない。** 学ぶのは「どの候補がどの候補より良いか」という
    相対順序だけで、選択そのもの (argmin) は損失の外側 (推論・regret 計測) にしか
    現れない。これが「探索は学習対象ではない」(SPEC) を損失の形で表したもの。

    対象ブロック: cov > min_cov (十分マスクを含む) かつ有効候補が2つ以上。
    ブロックごとにコストを有効候補の平均で正規化する (`Cn = C / mean(C_valid)`)。
    こうしないと、コストの絶対スケールがタイルごと・ブロックごとに違うため、
    margin (絶対値 0.1) が意味を持たない。

    ペア (i,j) は同一ブロック内の有効候補どうしで `err_i < order_ratio * err_j`
    (= i が明確に良い) となるものだけを対象にする。誤差がほぼ同じ (どちらでも
    良い) 候補どうしに強制的な順序を押し付けると、無用なノイズを学習してしまう。
    1タイルあたり最大 `max_pairs` ペアをランダムに間引く。

    loss = mean( relu(margin + Cn_i - Cn_j) )   (i が良い方)

    戻り値: 対象ペアが無ければ None (呼び出し側は step をスキップする)。
    """
    N, h, w = cost.shape
    blk = (cov > min_cov) & (usable.sum(0) >= 2)              # (h,w)
    if not bool(blk.any()):
        return None

    valid_cost = torch.where(usable, cost, torch.zeros_like(cost))
    n_valid = usable.to(cost.dtype).sum(0).clamp(min=1)
    mean_valid = valid_cost.sum(0) / n_valid                   # (h,w)
    Cn = cost / (mean_valid[None] + 1e-8)                      # (N,h,w)

    both_usable = usable[:, None] & usable[None, :]            # (N,N,h,w)
    order_ok = err[:, None] < order_ratio * err[None, :]       # (N,N,h,w)
    not_self = ~torch.eye(N, dtype=torch.bool, device=cost.device)[:, :, None, None]
    cand = both_usable & order_ok & not_self & blk[None, None]

    idx = cand.nonzero(as_tuple=False)                         # (P,4) -> i,j,y,x
    if idx.shape[0] == 0:
        return None
    if idx.shape[0] > max_pairs:
        pick = rng.choice(idx.shape[0], size=max_pairs, replace=False)
        idx = idx[torch.from_numpy(pick)]

    i, j, y, x = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    ci = Cn[i, y, x]
    cj = Cn[j, y, x]
    return F.relu(margin + ci - cj).mean()


def regret(cost: torch.Tensor, err: torch.Tensor, usable: torch.Tensor) -> float:
    """選んだ候補の実誤差 / 最良候補の実誤差。1.0 が神託 (v1 と同じ定義)。"""
    big = torch.full_like(cost, BIG)
    c = torch.where(usable, cost, big)
    e = torch.where(usable, err, big)
    pick = c.argmin(0)
    got = err.gather(0, pick[None])[0]
    best = e.min(0).values
    ok = torch.isfinite(best) & (best > 1e-9) & usable.any(0)
    if not bool(ok.any()):
        return float("nan")
    return float((got[ok] / best[ok]).median())


# --- キャッシュ読み出し ------------------------------------------------------

class Cache:
    def __init__(self, path: str, device: str):
        d = np.load(path)
        self.imgs = d["imgs"]
        self.queries = d["queries"]
        self.masks = d["masks"]
        self.off_idx = d["off_idx"]
        self.errs = d["errs"]
        self.covs = d["covs"]
        self.vfracs = d["vfracs"]
        self.offsets = d["offsets"]
        self.block = int(d["block"])
        self.scale = float(d["img_scale"])
        self.device = device
        self.n = len(self.queries)
        self.tile = self.queries.shape[1]

    def get(self, i: int):
        q = torch.from_numpy(
            self.queries[i].astype(np.float32) / self.scale
        ).permute(2, 0, 1)[None].to(self.device)
        m = np.unpackbits(self.masks[i])[:self.tile ** 2].reshape(
            self.tile, self.tile).astype(np.float32)
        mask = torch.from_numpy(m[None, None]).to(self.device)
        err = torch.from_numpy(self.errs[i]).to(self.device)       # (N,h,w)
        cov = torch.from_numpy(self.covs[i]).to(self.device)       # (h,w)
        vf = torch.from_numpy(self.vfracs[i]).to(self.device)      # (N,h,w)
        offs = [tuple(self.offsets[j]) for j in self.off_idx[i]]
        return q, mask, err, cov, vf, offs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/matcher2")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--overfit", type=int, default=0,
                    help=">0 なら先頭 N タイルだけで過学習テスト")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="cache 内の末尾何割を val に回すか (overfit 時は無効)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cache = Cache(a.cache, a.device)

    if a.overfit:
        train_ids = list(range(min(a.overfit, cache.n)))
        val_ids: list[int] = []
    else:
        n_val = max(1, int(round(cache.n * a.val_frac)))
        # **末尾 10% を cache 内 val に使う。** キャッシュ自体は既に train shard
        # のみから作られている (data/cache_candidates2.py) ので、ここでの分割は
        # 「学習に使うタイル」と「学習中の regret 監視に使うタイル」を分けるだけ。
        val_ids = list(range(cache.n - n_val, cache.n))
        train_ids = list(range(cache.n - n_val))
    p(f"キャッシュ {cache.n} タイル / 学習 {len(train_ids)} / cache内val {len(val_ids)} / "
      f"ブロック {cache.block}px")

    net = FeatureNet().to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    w0 = {k: v.detach().clone() for k, v in net.named_parameters()}
    p(f"FeatureNet params: {sum(q.numel() for q in net.parameters()):,}")

    rng = np.random.default_rng(a.seed)

    def cache_val_regret() -> float:
        if not val_ids:
            return float("nan")
        rs = []
        with torch.no_grad():
            for i in val_ids:
                q, mask, err, cov, vf, offs = cache.get(i)
                cost = block_costs(q, mask, offs, net, cache.block)
                usable = (vf > MIN_VALID) & (cov > MIN_COV)[None]
                r = regret(cost, err, usable)
                if np.isfinite(r):
                    rs.append(r)
        return float(np.median(rs)) if rs else float("nan")

    hist = []
    best_regret = float("inf")
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        i = train_ids[int(rng.integers(len(train_ids)))]
        q, mask, err, cov, vf, offs = cache.get(i)
        cost = block_costs(q, mask, offs, net, cache.block)
        usable = (vf > MIN_VALID) & (cov > MIN_COV)[None]

        loss = pairwise_hinge_loss(cost, err, usable, cov, rng)
        if loss is None:
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps:
            with torch.no_grad():
                # **named_parameters() のみ。** buffers (BatchNorm 統計など) を
                # 含めると「統計の暴走」を「パラメータが動いている」と誤読する
                # (stage_a の教訓)。FeatureNet に BatchNorm は無いが慣習として踏襲。
                dw = max(float((q_ - w0[k]).abs().max())
                         for k, q_ in net.named_parameters())
                r = cache_val_regret()
            el = time.perf_counter() - t0
            rec = dict(step=step, loss=float(loss.detach()), regret=r,
                       dw=dw, sec_per_step=el / step)
            hist.append(rec)
            p(f"  {step:>5}/{a.steps}  loss {rec['loss']:.5f}  "
              f"regret {r:6.3f}  |Δw|max {dw:.3e}  {rec['sec_per_step']*1000:.0f}ms/step")

            # **cache-val regret が最良のときだけ best.pt を更新する。**
            # 損失最小ではなく「選択の悪さ」が最良のチェックポイントを残す
            # (stage_a で loss 最小と実効性能が乖離した教訓と同じ理由)。
            if np.isfinite(r) and r < best_regret:
                best_regret = r
                os.makedirs(a.out, exist_ok=True)
                torch.save({"model": net.state_dict(), "step": step, "regret": r},
                           os.path.join(a.out, "best.pt"))

        if step % 1000 == 0 or step == a.steps:
            os.makedirs(a.out, exist_ok=True)
            torch.save({"model": net.state_dict(), "step": step},
                       os.path.join(a.out, "last.pt"))
            with open(os.path.join(a.out, "history.jsonl"), "w") as f:
                for rec in hist:
                    f.write(json.dumps(rec) + "\n")

    if len(hist) >= 2:
        p(f"\n=== 学習前後 ===")
        p(f"  loss    {hist[0]['loss']:.5f} -> {hist[-1]['loss']:.5f}")
        p(f"  regret  {hist[0]['regret']:.3f} -> {hist[-1]['regret']:.3f} "
          f"(最良 {best_regret:.3f}, 神託 1.00)")
        p(f"  |Δw|max {hist[-1]['dw']:.3e} "
          f"({'パラメータが動いている' if hist[-1]['dw'] > 1e-5 else '**動いていない**'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
