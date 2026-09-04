"""P3 事後診断 — 学習済み照合特徴がなぜ下流で悪化したかを切り分ける。

実測: cache-val regret は 1.481 -> 1.236 と改善したのに、
固定 val ベンチの PSNR は 16.47 -> 14.37 dB と 2.1 dB 悪化した。
代理指標と本番指標が逆を向いている。原因候補は2つ:

  H1 候補集合シフト
     学習はキャッシュした 64 候補**の中の順位**しか見ていない。
     推論は 2400 候補すべてで argmin を取る。学習が触れなかった
     2336 候補のどれかに不当に低いコストが付いていれば、順位学習が
     いくら正しくても argmin は壊れる。
  H2 コスト自体の劣化
     64 候補の中でも学習後のほうが悪い (単なる過学習・実装差)。

判別: 同じブロックについて
     - 64 候補に限定した argmin の真の誤差 (subset regret)
     - 全 2400 候補の argmin の真の誤差 (full regret)
  を手作り特徴と学習特徴の両方で測る。
  H1 なら「subset は学習が勝ち、full は学習が負け」になる。

  pixi run python eval/diagnose3.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.benchmark import load_shard_tiles                     # noqa: E402
from masks import sample_mask                                   # noqa: E402
from nagi_inpaint2 import stage_b_search as sbs                 # noqa: E402
from nagi_inpaint2.stage_a import StageA, make_query            # noqa: E402
from nagi_inpaint2.stage_b import FeatureNet                    # noqa: E402
from nagi_inpaint2.stage_b_simple import _shift                 # noqa: E402

DEV = "mps"
STAGE_A = "checkpoints/stage_a20k/last.pt"
MATCHER = "checkpoints/matcher2/best.pt"
BLOCK = 32
N_CASES = 10


def p(*a):
    print(*a, flush=True)


def true_err_all(x, mk, offs):
    """全オフセットの真のブロック誤差 (1,K,h,w)。無効ソースは BIG。"""
    src = 1.0 - mk
    blk = []
    for oy, ox in offs:
        vs = _shift(src, oy, ox)
        e = ((_shift(x, oy, ox) - x) ** 2).mean(1, keepdim=True)
        e = torch.where(vs > 0.5, e, torch.full_like(e, 1e6))
        blk.append(F.avg_pool2d(e * mk, BLOCK))
    return torch.cat(blk, 1) / (F.avg_pool2d(mk, BLOCK) + 1e-8)


def main():
    spec = json.load(open("cache/benchmark_val.json"))
    cases = spec["cases"][:N_CASES]
    sa = StageA().to(DEV)
    sa.load_state_dict(torch.load(STAGE_A, map_location=DEV)["model"])
    sa.eval()
    net = FeatureNet().to(DEV)
    mck = torch.load(MATCHER, map_location=DEV)
    net.load_state_dict(mck["model"])
    net.eval()
    p(f"matcher: {MATCHER} (step {mck.get('step')}, cache-regret {mck.get('regret')})")

    by_shard = {}
    for c in cases:
        by_shard.setdefault(c["shard"], []).append(c)

    rows = []
    rng = np.random.default_rng(0)
    for shard, cs in by_shard.items():
        tiles = {k: (im, sm) for k, im, sm, _ in
                 load_shard_tiles(os.path.join("shards", shard), stride=1)}
        for c in cs:
            if c["key"] not in tiles:
                continue
            img, _ = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(DEV)
            mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(DEV)
            with torch.inference_mode():
                q = make_query(sa, x, mk)
                _, offs, C_h = sbs.solve_blocks(x, mk, query=q)
                _, _, C_l = sbs.solve_blocks(x, mk, feat=net, query=q)
                E = true_err_all(x, mk, offs)
                cov = F.avg_pool2d(mk, BLOCK)
                inb = (cov[0, 0] > 0.05)
                if not inb.any():
                    continue
                # キャッシュと同じ候補構成: 40 ランダム + 手作りコスト上位 24
                K = len(offs)
                rnd = rng.permutation(K)[:40]
                rank_h = C_h[0, :, inb].mean(1).argsort()[:24].cpu().numpy()
                sub = np.unique(np.concatenate([rnd, rank_h]))
                subt = torch.from_numpy(sub).to(DEV)

                best = E[0, :, inb].min(0).values                # 神託 (全候補)
                def sel(C, idx=None):
                    Cc = C if idx is None else C[:, idx]
                    Ec = E if idx is None else E[:, idx]
                    j = Cc[0, :, inb].argmin(0)
                    return Ec[0, :, inb].gather(0, j[None])[0]
                r = dict(
                    full_h=float((sel(C_h) / (best + 1e-9)).median()),
                    full_l=float((sel(C_l) / (best + 1e-9)).median()),
                    sub_h=float((sel(C_h, subt) / (best + 1e-9)).median()),
                    sub_l=float((sel(C_l, subt) / (best + 1e-9)).median()),
                    # 選ばれたオフセットの多様性 (崩壊の検出)
                    nuniq_h=int(C_h[0, :, inb].argmin(0).unique().numel()),
                    nuniq_l=int(C_l[0, :, inb].argmin(0).unique().numel()),
                    nblk=int(inb.sum()))
                rows.append(r)
            torch.mps.empty_cache()

    p(f"\nケース {len(rows)} 件 / regret = 選択の真誤差 / 全候補中の最良 (1.0 が神託)\n")
    p(f"{'':<26}{'手作り':>10}{'学習':>10}")
    for k, lab in (("sub", "64候補内 regret"), ("full", "全2400候補 regret")):
        a = np.median([r[k + "_h"] for r in rows])
        b = np.median([r[k + "_l"] for r in rows])
        p(f"  {lab:<24}{a:>10.3f}{b:>10.3f}")
    a = np.median([r["nuniq_h"] for r in rows])
    b = np.median([r["nuniq_l"] for r in rows])
    n = np.median([r["nblk"] for r in rows])
    p(f"  {'選択オフセット種類数':<22}{a:>10.0f}{b:>10.0f}   (ブロック数 {n:.0f})")

    sh = np.median([r["sub_h"] for r in rows]); sl = np.median([r["sub_l"] for r in rows])
    fh = np.median([r["full_h"] for r in rows]); fl = np.median([r["full_l"] for r in rows])
    p("")
    if sl < sh and fl > fh:
        p("  => H1 候補集合シフト。学習は候補内の順位を改善したが、"
          "学習が見なかった候補に低コストを与えて全探索の argmin を壊している。")
        p("     処方: 学習ループ内で現行モデルの全探索 argmin を候補に加える"
          "(hard negative mining)。")
    elif sl > sh:
        p("  => H2 候補内でも劣化。学習設定そのものを疑う。")
    else:
        p("  => どちらでもない。別の要因 (前処理・正規化の不一致) を疑う。")


if __name__ == "__main__":
    main()
