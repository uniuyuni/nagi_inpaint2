"""LaMa と我々の出力を目視で比較する画像を作る (SPEC §15.3)。

**なぜ必要か**: 一連の測定で、PSNR がこの用途の品質を代表していないことが
分かった。平均化するほど PSNR は上がるのに、粒状は 1.145 -> 0.505 と失われ
写真としては劣化する。数値だけでは「どちらが良い写真か」を決められない。

マスクは細く小さい (電線・ゴミ、面積中央値 1.5%) ので、512px 全体を並べても
差が見えない。**マスクの外接矩形を中心に拡大した切り出し**を並べる。

各行: 元画像 | 入力 (穴を灰色で表示) | LaMa | 我々 (StageF+コピー)

  pixi run python eval/compare_visual.py --stage-f checkpoints/stage_f96/last.pt
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

from eval.benchmark import load_shard_tiles                     # noqa: E402
from eval.metrics import report                                 # noqa: E402
from masks import sample_mask                                   # noqa: E402
from nagi_inpaint2 import stage_b_search as sbs                 # noqa: E402
from nagi_inpaint2 import stage_d                                # noqa: E402
from nagi_inpaint2.stage_f import StageF                         # noqa: E402
from nagi_inpaint2.stage_fw import HypothesisQuery, load_stage_fw  # noqa: E402
from nagi_inpaint2.stage_fw2 import load_stage_fw2                # noqa: E402


def p(*a):
    print(*a, flush=True)


def to_u8(x: np.ndarray) -> np.ndarray:
    """知覚空間 (§3.3) の値を表示用 8bit にする。1.2 で割るのは既存流儀に合わせる。"""
    return (np.clip(x / 1.2, 0, 1) * 255).astype(np.uint8)


def crop_box(mask: np.ndarray, size: int, pad: int = 48):
    """マスクの外接矩形を中心に正方形で切り出す範囲を返す。

    細いマスクは全体像では見えないので、**必ず拡大して見せる**。
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0, size
    cy, cx = int((ys.min() + ys.max()) / 2), int((xs.min() + xs.max()) / 2)
    half = max(int(max(ys.max() - ys.min(), xs.max() - xs.min()) / 2) + pad, 64)
    half = min(half, size // 2)
    y0 = int(np.clip(cy - half, 0, size - 2 * half))
    x0 = int(np.clip(cx - half, 0, size - 2 * half))
    return y0, x0, 2 * half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="cache/bench_use_val.json")
    ap.add_argument("--stage-f", default="checkpoints/stage_f96/last.pt")
    ap.add_argument("--stage-fw", default=None,
                    help="StageFW (勝者総取り学習) の checkpoint。指定時は StageF の"
                         "代わりに使う")
    ap.add_argument("--hyp", choices=("first", "oracle", "copy"), default="first",
                    help="StageFW の K 個の仮説から1つを選ぶ方法。first=仮説0 "
                         "(製品経路)、oracle=GTに最も近い仮説 (上限測定専用)、"
                         "copy=Stage Bコピー結果に最も近い仮説")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--feather", type=float, default=0.0)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--d0", type=float, default=4.0)
    ap.add_argument("--d1", type=float, default=16.0)
    ap.add_argument("--band", action="store_true",
                    help="ラプラシアンピラミッドで周波数帯ごとに base(StageF)/copy を"
                         "混ぜる (band_composite)。32px 格子と貼ったブロックの浮きへの対処")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--cross", type=int, default=2)
    ap.add_argument("--photo", action="store_true",
                    help="コピー結果に photometric_align を適用し、ブロックの"
                         "明度の浮きを加算補正で消す")
    ap.add_argument("--grain", action="store_true",
                    help="Stage D の粒状注入を有効にする")
    ap.add_argument("--kind", default=None, help="この種別だけ比較する")
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--thumb", type=int, default=300)
    ap.add_argument("--out", default="out/compare")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from eval.lama_baseline import Lama
    spec = json.load(open(a.spec))
    if a.stage_fw:
        # StageF の代わりに StageFW/StageFW2 を使う。K 個の仮説から --hyp の方法で
        # 1つ選び、StageF と同じ (N,3,H,W) の呼び出し規約に揃えて既存経路に
        # そのまま流す。checkpoint の "arch" キーで判別する (無ければ StageFW)。
        _arch = torch.load(a.stage_fw, map_location="cpu").get("arch", "fw")
        if _arch == "fw2":
            fw_model, fwck = load_stage_fw2(a.stage_fw, a.device)
        else:
            fw_model, fwck = load_stage_fw(a.stage_fw, a.device)
        sf = HypothesisQuery(fw_model, a.hyp)
        p(f"Stage FW ({_arch}): {a.stage_fw} (step {fwck.get('step')}, "
          f"base {fwck.get('base')}, k {fw_model.k}, hyp={a.hyp})")
    else:
        fck = torch.load(a.stage_f, map_location=a.device)
        base = int(fck.get("base") or fck["model"]["out.weight"].shape[1])
        sf = StageF(base=base).to(a.device)
        sf.load_state_dict(fck["model"])
        sf.eval()
        p(f"Stage F: {a.stage_f} (step {fck.get('step')}, base {base})")
    lama = Lama(device=a.device)

    by_shard = {}
    sel = [c for c in spec["cases"] if not a.kind or c["kind"] == a.kind]
    for c in sel[:a.cases]:
        by_shard.setdefault(c["shard"], []).append(c)

    os.makedirs(a.out, exist_ok=True)
    rows, n = [], 0
    for shard, cases in by_shard.items():
        tiles = {k: (im, sm) for k, im, sm, _ in
                 load_shard_tiles(os.path.join("shards", shard), stride=1)}
        for c in cases:
            if c["key"] not in tiles:
                continue
            img, satm = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(a.device)
            mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(a.device)

            fl = lama(img, mask)
            out_lama = np.where(mask[..., None], fl, img)

            with torch.inference_mode():
                q = torch.where(mk > 0.5, sf(x, mk), x)
                o = sbs.inpaint(x, mk, query=q, topk=a.topk, feather=a.feather,
                                adaptive=a.adaptive, d0=a.d0, d1=a.d1,
                                band=a.band, levels=a.levels, cross=a.cross,
                                photo=a.photo)[0]
                o = o.permute(1, 2, 0).cpu().numpy()
            d = stage_d.harmonize(o, img, mask, valid=~satm,
                                  rng=np.random.default_rng(0),
                                  do_color=False, do_poisson=True, do_grain=a.grain)
            out_ours = np.where(mask[..., None], d, img)

            masked = img.copy()
            masked[mask] = 0.45                       # 穴を灰色で示す

            y0, x0, s = crop_box(mask, c["tile"])
            sl = (slice(y0, y0 + s), slice(x0, x0 + s))
            pl = report(out_lama, img, mask, gt=img)["psnr"]
            po = report(out_ours, img, mask, gt=img)["psnr"]
            rows.append((f"{c['kind']} 面積{c['area']*100:.1f}%  "
                         f"LaMa {pl:.1f}dB / 我々 {po:.1f}dB",
                         img[sl], masked[sl], out_lama[sl], out_ours[sl]))
            n += 1
            p(f"  {n}: {c['kind']:6s} LaMa {pl:5.1f} / 我々 {po:5.1f}")

    # 4列の対比シート
    t, gap, lab = a.thumb, 8, 22
    W = t * 4 + gap * 3
    H = sum(t + lab for _ in rows)
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    y = 0
    for title, gt, msk, lm, ou in rows:
        for i, arr in enumerate((gt, msk, lm, ou)):
            im = Image.fromarray(to_u8(arr)).resize((t, t), Image.NEAREST)
            sheet.paste(im, (i * (t + gap), y))
        y += t
        dr.text((4, y + 4), title, fill=(235, 235, 235))
        y += lab
    dr.text((4, 2), "元画像", fill=(255, 220, 120))
    dr.text((t + gap + 4, 2), "入力(穴)", fill=(255, 220, 120))
    dr.text((2 * (t + gap) + 4, 2), "LaMa", fill=(255, 220, 120))
    dr.text((3 * (t + gap) + 4, 2), "我々", fill=(255, 220, 120))
    nm = (f"cmp_{a.kind or 'all'}_k{a.topk}"
          f"{'_A' if a.adaptive else ''}{'_PY' if a.band else ''}"
          f"{'_PH' if a.photo else ''}{'_G' if a.grain else ''}.png")
    path = os.path.join(a.out, nm)
    sheet.save(path)
    p(f"\n-> {path}  ({len(rows)} ケース)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
