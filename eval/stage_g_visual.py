"""Stage G の目視確認 (G1 の主たる判定手段)。

**なぜ目視が主たる判定手段か**: `nagi_inpaint2/stage_g.py` の docstring にある
とおり、この設計が解こうとしている問題は「PSNR は動かないのに人が見ると
ぼけた塊に見える」というものだった (盲検A/B 0勝16敗4分)。`train/train_stage_g.py`
の `[grain]` ログは一次スクリーニングだが、grain 比という単一のスカラーに
本当に "ぼけが消えたか" が正しく反映されているかは、結局 **目で見て確認する
必要がある** (これも過去の教訓: `eval/quickval.py` 参照。学習側の代理指標と
最終的な見え方が乖離した例がこのセッションで複数回あった)。

**必ず人物マスクのケースで確認する** — `stage_g.py` の docstring にあるとおり
細いマスクでは grain が nan になり判定できない。既定の `--spec` は
`cache/bench_person_val.json` (人物サイズの silhouette マスクで構成された
保留ケース集合、`eval/quickval.py` と同じ)。

各行: 元画像 | 入力 (穴を灰色で表示) | StageF単体 | StageF+StageG

  pixi run python eval/stage_g_visual.py --stage-g checkpoints/stage_g/last.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.benchmark import load_shard_tiles              # noqa: E402
from eval.compare_visual import crop_box, to_u8           # noqa: E402
from masks import sample_mask                             # noqa: E402
from nagi_inpaint2.stage_g import StageG, sample_residual  # noqa: E402
from train.train_stage_g import compute_r_cond, load_frozen_stage_f  # noqa: E402
from train.train_stage_g2 import (compute_r_cond_scaled,   # noqa: E402
                                  sample_residual_scaled)


def p(*a):
    print(*a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="cache/bench_person_val.json")
    ap.add_argument("--stage-f", default="checkpoints/stage_f96/last.pt")
    ap.add_argument("--stage-g", default="checkpoints/stage_g/last.pt")
    ap.add_argument("--scaled", action="store_true",
                    help="残差の局所スケール正規化版 (`train_stage_g2.py` の"
                         "checkpoint) を読み、`sample_residual_scaled` で"
                         "サンプリングする。それ以外の挙動 (レイアウト・"
                         "出力先など) は変えない")
    ap.add_argument("--use-copy-cond", type=int, default=1,
                    help="Stage B の検索パッチ残差を条件に含めるか"
                         "(`train_stage_g.py` の学習設定と揃えること)")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8,
                    help="sample_residual のオイラー法ステップ数")
    ap.add_argument("--seed", type=int, default=0,
                    help="sample_residual の再現性を取るための generator の種")
    ap.add_argument("--thumb", type=int, default=300)
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--out", default="out/g")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    cases = spec["cases"][:a.cases]

    sf = load_frozen_stage_f(a.stage_f, a.device)

    gck = torch.load(a.stage_g, map_location=a.device)
    g_base = int(gck.get("base", 32))
    cond_ch = int(gck.get("cond_ch", 9 if a.scaled else 8))
    g = StageG(base=g_base, cond_ch=cond_ch).to(a.device)
    g.load_state_dict(gck["model"])
    g.eval()
    p(f"  StageG: {a.stage_g} (step {gck.get('step')}, base {g_base}, "
      f"cond_ch={cond_ch}, scaled={a.scaled}, steps={a.steps})")

    by_shard: dict[str, list] = {}
    for c in cases:
        by_shard.setdefault(c["shard"], []).append(c)

    os.makedirs(a.out, exist_ok=True)
    rows = []
    n = 0
    for shard, cs in by_shard.items():
        tiles = {k: (im, sm) for k, im, sm, _ in
                 load_shard_tiles(os.path.join(a.shards, shard), stride=1)}
        for c in cs:
            if c["key"] not in tiles:
                continue
            img, _satm = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(a.device)
            mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(a.device)

            gen = torch.Generator().manual_seed(a.seed)
            if a.scaled:
                r, cond, sf_out, scale = compute_r_cond_scaled(sf, x, mk, bool(a.use_copy_cond))
            else:
                r, cond, sf_out = compute_r_cond(sf, x, mk, bool(a.use_copy_cond))
            with torch.no_grad():
                if a.scaled:
                    r_hat = sample_residual_scaled(g, cond, mk, scale, steps=a.steps,
                                                   generator=gen)
                else:
                    r_hat = sample_residual(g, cond, mk, steps=a.steps, generator=gen)
                out_f = torch.where(mk > 0.5, sf_out, x)
                out_g = torch.where(mk > 0.5, sf_out + r_hat, x)

            out_f_np = out_f[0].permute(1, 2, 0).cpu().numpy()
            out_g_np = out_g[0].permute(1, 2, 0).cpu().numpy()

            masked = img.copy()
            masked[mask] = 0.45                       # 穴を灰色で示す

            y0, x0, s = crop_box(mask, c["tile"])
            sl = (slice(y0, y0 + s), slice(x0, x0 + s))
            rows.append((f"{c['kind']} 面積{c['area']*100:.1f}%",
                        img[sl], masked[sl], out_f_np[sl], out_g_np[sl]))
            n += 1
            p(f"  {n}: {c['key']} ({c['kind']})")

    if not rows:
        p("該当ケースが無い (spec/shards のパスを確認)")
        return 1

    # 4列の対比シート (`eval/compare_visual.py` と同じレイアウト)
    t, gap, lab = a.thumb, 8, 22
    W = t * 4 + gap * 3
    H = sum(t + lab for _ in rows)
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    y = 0
    for title, gt, msk, of, og in rows:
        for i, arr in enumerate((gt, msk, of, og)):
            im = Image.fromarray(to_u8(arr)).resize((t, t), Image.NEAREST)
            sheet.paste(im, (i * (t + gap), y))
        y += t
        dr.text((4, y + 4), title, fill=(235, 235, 235))
        y += lab
    dr.text((4, 2), "元画像", fill=(255, 220, 120))
    dr.text((t + gap + 4, 2), "入力(穴)", fill=(255, 220, 120))
    dr.text((2 * (t + gap) + 4, 2), "StageF単体", fill=(255, 220, 120))
    dr.text((3 * (t + gap) + 4, 2), "StageF+StageG", fill=(255, 220, 120))
    path = os.path.join(a.out, f"stage_g_cases{len(rows)}_steps{a.steps}.png")
    sheet.save(path)
    p(f"\n-> {path}  ({len(rows)} ケース)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
