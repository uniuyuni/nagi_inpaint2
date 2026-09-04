"""固定ベンチマーク — 再現可能な比較基盤 (SPEC §9)。

**このスクリプトを作った理由**

これまでのベンチマークには2つの欠陥があった。

  1. **学習データが混入していた。** `sample_tiles("shards", ...)` は全 shard から
     標本を取るため、Stage A が学習に使ったタイルが含まれていた。
     長く学習したモデルほど有利に出る。実測でチェックポイントの順位が入れ替わった:

         全shard標本 (漏洩あり)   30k-last 28.30 (1位) / 20k-last 26.92 (2位)
         保留shardのみ           20k-last 18.15 (1位) / 30k-last 17.95 (3位)

     LaMa との比較では影響しない (LaMa はこのデータで学習していない) が、
     自前モデル同士の比較・チェックポイント選択・「学習が効いたか」の判断は無効になる。

  2. **毎回サンプリングし直していた。** ケース数 24 と少なく、実行ごとに
     対象が変わるため、小さな差が測れなかった。

本スクリプトは**保留 shard から固定のケース集合**を作り、JSON に保存して再利用する。
一度作れば以降は同じケースで比較でき、乱数による揺れが入らない。

**保留 shard の制約 (既知の弱点)**: 現在 5 本しかなく、内容の偏りがある
(ぼかし基準で学習側より 1.47倍難しい)。次に Stage A を学習する際は、
**先に保留を 20 本程度へ拡大**してから学習すべき。今それをすると
現行モデルが学習済みの shard を保留に含めてしまうので、いまは 5 本のまま使う。

  pixi run python eval/benchmark.py --build          # ケース集合を作る
  pixi run python eval/benchmark.py --stage-a checkpoints/stage_a20k/last.pt
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tarfile

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.metrics import _gray, _highpass, report        # noqa: E402
from masks import sample_mask                            # noqa: E402
from nagi_inpaint2 import stage_b_search as sbs          # noqa: E402
from nagi_inpaint2 import stage_d                        # noqa: E402
from nagi_inpaint2.color import byte_unshuffle, dequantize   # noqa: E402
from nagi_inpaint2.stage_a import StageA, make_query     # noqa: E402
from nagi_inpaint2.stage_f import StageF                 # noqa: E402
from nagi_inpaint2.stage_fw import HypothesisQuery, load_stage_fw  # noqa: E402
from nagi_inpaint2.stage_fw2 import load_stage_fw2                # noqa: E402
from train.dataset import list_shards, split_shards      # noqa: E402

SPEC = "cache/benchmark.json"


def p(*a):
    print(*a, flush=True)


def hf_ratio(x: np.ndarray) -> float:
    g = _gray(x)
    return float(_highpass(g).std() / (g.std() + 1e-6))


def load_shard_tiles(path: str, stride: int = 1):
    d = zstd.ZstdDecompressor()
    out = []
    with tarfile.open(path) as tf:
        keys = sorted(n[:-5] for n in tf.getnames() if n.endswith(".json"))
        for k in keys[::stride]:
            meta = json.loads(tf.extractfile(k + ".json").read())
            img = byte_unshuffle(
                d.decompress(tf.extractfile(k + ".img.u16.zst").read()),
                tuple(meta["shape"]))
            sat = np.load(io.BytesIO(
                d.decompress(tf.extractfile(k + ".sat.npy.zst").read())))
            satm = np.unpackbits(sat)[:meta["tile"] ** 2].reshape(
                meta["tile"], meta["tile"]).astype(bool)
            out.append((k, dequantize(img, meta["q_scale"], meta["q_offset"]),
                        satm, meta))
    return out


def build(shard_dir: str, out: str, n_cases: int, hf_pct: float, seed: int,
          kinds: list[str] | None = None):
    """保留 shard から固定のケース集合を作る。"""
    _, va = split_shards(list_shards(shard_dir))
    p(f"保留 shard {len(va)} 本からケースを構築")
    cands = []
    for s in va:
        for key, img, satm, meta in load_shard_tiles(s, stride=2):
            cands.append((s, key, hf_ratio(img), meta["tile"]))
    hfs = np.array([c[2] for c in cands])
    thr = np.percentile(hfs, hf_pct)
    keep = [c for c in cands if c[2] >= thr]
    p(f"  候補 {len(cands):,} タイル -> 高周波上位 {100-hf_pct:.0f}% で {len(keep):,}")

    rng = np.random.default_rng(seed)
    sel = rng.permutation(len(keep))[:n_cases * 2]
    cases = []
    for i in sel:
        s, key, h, tile = keep[int(i)]
        mrng = np.random.default_rng(int(i) + 10000)
        # **マスク種別をベンチの目的に合わせて選ぶ。**
        # 当初は blob 固定だったが、SPEC の対象範囲は電線・ゴミ・小さな通行人・看板で
        # あり、blob (塊) はその中で最も不利かつ用途から外れた形状だった。
        # 用途ベンチでは wire/dust/stroke/rect を混ぜる。
        kd = str(mrng.choice(kinds)) if kinds else "blob"
        mask, mm = sample_mask(tile, mrng, kind=kd)
        if mask.sum() < 200:
            continue
        cases.append(dict(shard=os.path.basename(s), key=key, hf=round(h, 5),
                          mask_seed=int(i) + 10000, tile=tile,
                          area=mm["area"], kind=mm["kind"]))
        if len(cases) >= n_cases:
            break
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(shards=[os.path.basename(x) for x in va],
                       hf_pct=hf_pct, seed=seed, cases=cases), f, indent=1)
    p(f"  ケース {len(cases)} 件 -> {out}")
    p(f"  マスク面積 中央値 {np.median([c['area'] for c in cases])*100:.1f}%")


def bootstrap_median_ci(diffs: list[float], n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float]:
    """対応のある差の中央値のブートストラップ 95%信頼区間。

    **なぜ要るか**: 人物ベンチ 60ケースでの 0.0059 対 0.0052 という差が
    統計的に有意かを一度も確認していなかった。中央値だけを並べていると、
    雑音を改善だと誤読する事故が起きる。ケースを重複を許して再標本し、
    その中央値の分布から区間を作ることで、この差がどれだけ揺れうるかを見る。
    """
    v = np.asarray([d for d in diffs if np.isfinite(d)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.median(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def sign_test_pvalue(diffs: list[float]) -> float:
    """符号検定の両側 p 値 (0 との差)。scipy を増やさず `math.comb` で自前計算する。

    「同点 (差が厳密に0)」は検定から除外する (符号検定の標準的な扱い)。
    """
    v = [d for d in diffs if np.isfinite(d) and d != 0]
    n = len(v)
    if n == 0:
        return float("nan")
    k = sum(1 for d in v if d > 0)
    obs = math.comb(n, k) * (0.5 ** n)
    total = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (0.5 ** n)
        if pi <= obs * (1 + 1e-9):
            total += pi
    return float(min(1.0, total))


def paired_comparison(ours: list[dict], base: list[dict], key: str) -> dict:
    """`ours[i][key] - base[i][key]` (ケースごとの対応する差) の要約統計を返す。

    `ours` と `base` は同じケース列を同じ順序で処理した結果 (`run()` 内では
    どちらも `spec["cases"]` を同一順で辿って積んでいるので、インデックスが
    そのままケース対応になる)。戻り値: `n`(有限差の件数), `median`,
    `ci_lo`/`ci_hi` (ブートストラップ2000回), `pvalue` (符号検定)。
    """
    diffs = [a.get(key, float("nan")) - b.get(key, float("nan"))
             for a, b in zip(ours, base)]
    diffs = [d for d in diffs if np.isfinite(d)]
    lo, hi = bootstrap_median_ci(diffs)
    return dict(n=len(diffs),
               median=float(np.median(diffs)) if diffs else float("nan"),
               ci_lo=lo, ci_hi=hi, pvalue=sign_test_pvalue(diffs))


def _build_query(x, mk, img, mask, src, sa, sf, lama, device):
    """Stage B に渡すクエリ (穴を埋めた画像) を作る。

    `stagef` : 全解像度の学習モデルの出力。**現行の既定方針**。
               学習モデルは精度が出るがぼけ、コピーはテクスチャを持つが精度が出ない。
               構造を StageF、テクスチャをコピーに担わせる (SPEC §15)。
    `stagea` : 1/8 解像度の低周波補完。旧方針。比較用に残す。
    `lama`   : LaMa 出力の低周波。**上限測定専用** (R2)。製品経路ではない。
    """
    if src == "stagef":
        return torch.where(mk > 0.5, sf(x, mk), x)
    if src == "lama":
        f = lama(img, mask)
        return sbs.lowfreq_query(
            x, mk, torch.from_numpy(f.transpose(2, 0, 1)[None]).to(device))
    return make_query(sa, x, mk)


def run(shard_dir: str, spec_path: str, stage_a: str | None,
        with_lama: bool, device: str, grain: bool = False,
        dump: str | None = None, query_src: str = "stagef",
        topk: int = 1, tau_k: float = 0.5, patch: int = 7,
        stage_f: str | None = None,
        hf_only: bool = False, hf_sigma: float = 2.0, hf_alpha: float = 1.0,
        feather: float = 0.0,
        adaptive: bool = False, d0: float = 8.0, d1: float = 24.0,
        band: bool = False, levels: int = 5, cross: int = 2,
        photo: bool = False,
        lpips=None,
        stage_fw: str | None = None, hyp: str = "first"):
    """固定ベンチで LaMa と比較する。

    評価する経路は3つだけに絞ってある:
      我々(クエリなし)  Stage B 単体 (クエリ無し。参考値)
      我々(<query>+B)  クエリ + Stage B のコピー (+ Stage D)
      我々(StageF)     StageF 単体 (Stage B を通さない)

    **探索用のフラグは削除した。** 信頼度ゲート・照合コスト学習・近傍コヒーレンス・
    ブートストラップ再探索はいずれも実測で不採用になっている (eval/RESULTS.md)。
    それらを引数として残したまま探索を続けた結果、引数22個・分岐41個まで肥大し、
    CLI 配線由来のバグを3件出した (base 不整合・タグ未登録・経路ゲートの取り違え)。
    採否が決まった機能は消す。実装自体は nagi_inpaint2/stage_b_search.py に残っている。

    `lpips` (`eval.perceptual.Perceptual` インスタンス、既定 None) を渡すと、
    各手法の出力に対して知覚距離も測り `report()` の結果 dict に "lpips" として
    後から代入する。None のままなら一切計算しない (`--lpips` 未指定時の既存の
    出力・数値を変えないため)。

    `stage_fw` (`StageFW` の checkpoint パス) を渡すと、`StageF` の代わりに
    `StageFW` を使う。`hyp` でその K 個の仮説から1つを選ぶ方法を決める
    (`nagi_inpaint2.stage_fw.select_hypothesis` 参照): "first" は仮説0
    (想定される製品経路)、"oracle" は GT に最も近い仮説 (**上限測定専用**、
    手法名に "(oracle)" を付ける)、"copy" は Stage B のコピー結果に最も
    近い仮説 (GT を使わない実用的な選択)。
    """
    spec = json.load(open(spec_path))
    by_shard = {}
    for c in spec["cases"]:
        by_shard.setdefault(c["shard"], []).append(c)
    p(f"固定ベンチマーク {len(spec['cases'])} ケース / 保留 shard {len(spec['shards'])} 本")

    sa = None
    if stage_a:
        sa = StageA().to(device)
        ck = torch.load(stage_a, map_location=device)
        sa.load_state_dict(ck["model"])
        sa.eval()
        p(f"  Stage A: {stage_a} (step {ck.get('step')})")
    sf = None
    if stage_fw:
        # StageF の代わりに StageFW/StageFW2 を使う。K 個の仮説から `hyp` の方法で
        # 1つ選び、選んだ仮説を StageF と同じ (N,3,H,W) の呼び出し規約に揃えて
        # 既存経路 (_build_query, 単体の "我々(StageFW)" 経路) にそのまま流す。
        # **checkpoint の "arch" キーで StageFW / StageFW2 を判別する。** 古い
        # checkpoint (arch キーが無い) は StageFW とみなす。新しいフラグは
        # 追加しない (どちらの checkpoint でも --stage-fw 一つで読める)。
        _arch = torch.load(stage_fw, map_location="cpu").get("arch", "fw")
        if _arch == "fw2":
            fw_model, fwck = load_stage_fw2(stage_fw, device)
        else:
            fw_model, fwck = load_stage_fw(stage_fw, device)
        sf = HypothesisQuery(fw_model, hyp)
        p(f"  Stage FW ({_arch}): {stage_fw} (step {fwck.get('step')}, "
          f"base {fwck.get('base')}, k {fw_model.k}, hyp={hyp}, "
          f"params {sum(q.numel() for q in fw_model.parameters()):,})")
    elif stage_f:
        fck = torch.load(stage_f, map_location=device)
        # base は checkpoint から復元する (--base で変えられるため、既定値で作ると
        # 形が合わずロードに失敗する)。出力層 out.weight の形 (3, base, 3, 3) で決まる。
        base = int(fck.get("base") or fck["model"]["out.weight"].shape[1])
        sf = StageF(base=base).to(device)
        sf.load_state_dict(fck["model"])
        sf.eval()
        p(f"  Stage F: {stage_f} (step {fck.get('step')}, base {base}, "
          f"params {sum(q.numel() for q in sf.parameters()):,})")
    lama = None
    if with_lama:
        from eval.lama_baseline import Lama
        lama = Lama(device=device)

    # クエリ源が用意されていなければ Stage B 経路は走らせない
    have_q = {"stagef": sf is not None, "stagea": sa is not None,
              "lama": lama is not None}[query_src]
    ktag = f"+K{topk}" if topk >= 2 else ""
    ptag = f"+p{patch}" if patch != 7 else ""
    htag = "+HF" if hf_only else ""
    ftag = "+F" if feather > 0 else ""
    atag = "+A" if adaptive else ""
    ytag = "+PY" if band else ""
    htag2 = "+PH" if photo else ""
    if query_src == "stagef" and stage_fw:
        qtag = f"StageFW-{hyp}"
    else:
        qtag = {"stagef": "StageF", "lama": "LaMaLF", "stagea": "StageA"}[query_src]
    tag = f"我々({qtag}+B{ptag}{ktag}{htag}{ftag}{atag}{ytag}{htag2})"
    if hyp == "oracle" and stage_fw:
        # oracle は GT を見て仮説を選ぶ上限測定専用の診断値。製品経路と混同しない
        # よう、手法名に必ず "(oracle)" を付ける。
        tag += "(oracle)"

    # StageFW 使用時は単体経路のラベルも区別する (同じ "StageF" 表記だと
    # StageF/StageFW どちらの結果か分からなくなる)。
    standalone = "我々(StageFW)" if stage_fw else "我々(StageF)"

    names = (["LaMa"] if lama else []) + ["我々(クエリなし)"]
    names += [tag, tag + "+StageD"] if have_q else []
    # StageF/StageFW 単体も併記する (クエリとして使う場合は同じモデルの別経路として比較になる)
    names += [standalone, standalone + "+StageD"] if sf else []
    acc = {k: [] for k in names}
    percase = []
    kinds: dict[str, list[int]] = {}   # マスク種別ごとの行番号 (種別別集計用)

    for shard, cases in by_shard.items():
        tiles = {k: (im, sm, mt) for k, im, sm, mt
                 in load_shard_tiles(os.path.join(shard_dir, shard), stride=1)}
        for c in cases:
            if c["key"] not in tiles:
                continue
            img, satm, _ = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(device)
            mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
            rng0 = lambda: np.random.default_rng(0)

            def add(name, out):
                """マスク内だけ差し替えて評価し、Stage D 版も併せて積む。"""
                full = np.where(mask[..., None], out, img)
                acc[name].append(report(full, img, mask, gt=img))
                if lpips is not None:
                    acc[name][-1]["lpips"] = lpips(full, img, mask=mask)
                d = stage_d.harmonize(out, img, mask, valid=~satm, rng=rng0(),
                                      do_color=False, do_poisson=True, do_grain=grain)
                acc[name + "+StageD"].append(report(d, img, mask, gt=img))
                if lpips is not None:
                    acc[name + "+StageD"][-1]["lpips"] = lpips(d, img, mask=mask)

            kinds.setdefault(c["kind"], []).append(len(acc.get("LaMa", [])))
            if lama:
                f = lama(img, mask)
                full_lama = np.where(mask[..., None], f, img)
                acc["LaMa"].append(report(full_lama, img, mask, gt=img))
                if lpips is not None:
                    acc["LaMa"][-1]["lpips"] = lpips(full_lama, img, mask=mask)
            with torch.inference_mode():
                o0 = sbs.inpaint(x, mk)[0].permute(1, 2, 0).cpu().numpy()
            full0 = np.where(mask[..., None], o0, img)
            acc["我々(クエリなし)"].append(report(full0, img, mask, gt=img))
            if lpips is not None:
                acc["我々(クエリなし)"][-1]["lpips"] = lpips(full0, img, mask=mask)

            if have_q:
                with torch.inference_mode():
                    q = _build_query(x, mk, img, mask, query_src, sa, sf, lama, device)
                    o1 = sbs.inpaint(x, mk, query=q, topk=topk, tau_k=tau_k,
                                     patch=patch, hf_only=hf_only, hf_sigma=hf_sigma,
                                     hf_alpha=hf_alpha, feather=feather,
                                     adaptive=adaptive, d0=d0, d1=d1,
                                     band=band, levels=levels, cross=cross,
                                     photo=photo,
                                     )[0].permute(1, 2, 0).cpu().numpy()
                add(tag, o1)
                percase.append(dict(shard=c["shard"], key=c["key"],
                                    psnr=acc[tag + "+StageD"][-1].get("psnr")))
            if sf:
                # StageF/StageFW は全解像度を直接補完するので Stage B を経由しない。
                # マスク外にも生出力を残したまま harmonize に渡す契約 (§4.4)。
                with torch.inference_mode():
                    of = sf(x, mk)[0].permute(1, 2, 0).cpu().numpy()
                add(standalone, of)

    def med(rs, k):
        v = [x[k] for x in rs if np.isfinite(x.get(k, np.nan))]
        return float(np.median(v)) if v else float("nan")

    lpips_hdr = f"{'lpips':>8}" if lpips is not None else ""
    p(f"\n{'手法':<24}{'boundary':>10}{'grain':>8}{'psd':>9}{lpips_hdr}{'psnr':>8}{'勝率':>9}")
    base_r = acc.get("LaMa")
    for k, rs in acc.items():
        if not rs:
            continue
        w = ""
        if base_r and k != "LaMa":
            w = f"{sum(1 for a, b in zip(rs, base_r) if a['psnr'] > b['psnr'])}/{len(rs)}"
        lpips_col = f"{med(rs,'lpips'):>8.4f}" if lpips is not None else ""
        p(f"{k:<24}{med(rs,'boundary'):>10.3f}{med(rs,'grain'):>8.3f}"
          f"{med(rs,'psd'):>9.4f}{lpips_col}{med(rs,'psnr'):>8.2f}{w:>9}")
    # **マスク種別ごとの内訳。** 全体の中央値だけでは、どの形状で破綻しているかが
    # 分からない。実際 wire では勝ち rect では大敗する、と種別で挙動が逆転していた。
    if len(kinds) > 1 and acc.get("LaMa"):
        others = [k for k in acc if k != "LaMa" and acc[k]]
        hdr = f"\n{'種別':<8}{'n':>4}{'LaMa':>9}" + "".join(
            f"{k[-14:]:>16}" for k in others)
        p(hdr)
        for kd, rows_i in sorted(kinds.items()):
            line = (f"{kd:<8}{len(rows_i):>4}"
                    f"{np.median([acc['LaMa'][i]['psnr'] for i in rows_i]):>9.2f}")
            for k in others:
                v = [acc[k][i]['psnr'] for i in rows_i if i < len(acc[k])]
                line += f"{np.median(v) if v else float('nan'):>16.2f}"
            p(line)
    # **対応のある差の統計。** 既存の表は各手法の中央値しか出しておらず、
    # 「LaMa との差が雑音か本物か」を一度も検定していなかった (冒頭の背景を参照)。
    # 同じケース列を辿っているので `acc[k][i]` と `acc["LaMa"][i]` はケース対応
    # している。既存の表・数値は変えず、末尾に追加するだけにしてある。
    if base_r:
        keys_to_test = ["psnr"] + (["lpips"] if lpips is not None else [])
        for metric in keys_to_test:
            p(f"\n対 LaMa の差 ({metric}, 我々-LaMa, 符号検定 = 中央値0との差):")
            hdr = f"{'手法':<24}{'n':>4}{'中央値':>9}{'95%CI':>18}{'p値':>10}"
            p(hdr)
            for k, rs in acc.items():
                if k == "LaMa" or not rs:
                    continue
                stat = paired_comparison(rs, base_r, metric)
                if stat["n"] == 0:
                    continue
                ci = f"[{stat['ci_lo']:.3f}, {stat['ci_hi']:.3f}]"
                p(f"{k:<24}{stat['n']:>4}{stat['median']:>9.3f}{ci:>18}"
                  f"{stat['pvalue']:>10.4f}")
    if dump:
        json.dump(percase, open(dump, "w"), indent=1)
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--hf-pct", type=float, default=75.0)
    ap.add_argument("--kinds", default=None,
                    help="カンマ区切りのマスク種別。既定 (None) は blob 固定")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-a", default=None)
    ap.add_argument("--stage-f", default=None)
    ap.add_argument("--stage-fw", default=None,
                    help="StageFW (勝者総取り学習) の checkpoint。指定時は StageF の"
                         "代わりに使う")
    ap.add_argument("--hyp", choices=("first", "oracle", "copy"), default="first",
                    help="StageFW の K 個の仮説から1つを選ぶ方法。first=仮説0 "
                         "(製品経路)、oracle=GTに最も近い仮説 (上限測定専用)、"
                         "copy=Stage Bコピー結果に最も近い仮説")
    ap.add_argument("--query-src", choices=("stagef", "stagea", "lama"),
                    default="stagef")
    ap.add_argument("--no-lama", action="store_true")
    ap.add_argument("--grain", action="store_true")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--tau-k", type=float, default=0.5)
    ap.add_argument("--patch", type=int, default=7)
    ap.add_argument("--hf-only", action="store_true",
                    help="コピー結果を丸ごと使わず、高周波成分だけを query に移植する")
    ap.add_argument("--hf-sigma", type=float, default=2.0)
    ap.add_argument("--hf-alpha", type=float, default=1.0)
    ap.add_argument("--feather", type=float, default=0.0,
                    help="ブロック貼り付けの重なり羽化 (paste_feather の overlap)。"
                         "0 以下で無効 (従来の paste)。topk>=2 との併用は無視される")
    ap.add_argument("--adaptive", action="store_true",
                    help="穴の太さ (hole_distance) で copy と query (StageF) を"
                         "画素単位でブレンドする (adaptive_blend)。rect (太い矩形)"
                         "でコピーが StageF に負ける逆転を画素単位で解消する")
    ap.add_argument("--d0", type=float, default=8.0,
                    help="adaptive_blend: この距離以内は copy をそのまま使う")
    ap.add_argument("--d1", type=float, default=24.0,
                    help="adaptive_blend: この距離を超えたら完全に query に委ねる")
    ap.add_argument("--band", action="store_true",
                    help="ラプラシアンピラミッドで周波数帯ごとに base(query)/copy を"
                         "混ぜる (band_composite)。adaptive の代わりに使う。"
                         "32px 格子と貼ったブロックの浮きへの対処")
    ap.add_argument("--levels", type=int, default=5,
                    help="band_composite: ラプラシアンピラミッドの段数")
    ap.add_argument("--cross", type=int, default=2,
                    help="band_composite: この段より粗い帯は base、細かい帯は copy")
    ap.add_argument("--photo", action="store_true",
                    help="コピー結果に photometric_align を適用し、ブロックの"
                         "明度の浮きを加算補正で消す")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--lpips", action="store_true",
                    help="知覚指標 (LPIPS 相当, eval/perceptual.py) も計算する。"
                         "既定では計算しない (重みロードのコストと、この指標を"
                         "使わない既存の実行を遅くしないため)")
    a = ap.parse_args()

    if a.build:
        build(a.shards, a.spec, a.cases, a.hf_pct, a.seed,
              a.kinds.split(",") if a.kinds else None)
        return 0
    if not os.path.exists(a.spec):
        p(f"ケース集合が無い: {a.spec}  先に --build を実行すること")
        return 1
    lpips = None
    if a.lpips:
        from eval.perceptual import Perceptual
        lpips = Perceptual(device=a.device)
    run(a.shards, a.spec, a.stage_a, not a.no_lama, a.device, grain=a.grain,
        dump=a.dump, query_src=a.query_src, topk=a.topk, tau_k=a.tau_k,
        patch=a.patch, stage_f=a.stage_f, hf_only=a.hf_only,
        hf_sigma=a.hf_sigma, hf_alpha=a.hf_alpha, feather=a.feather,
        adaptive=a.adaptive, d0=a.d0, d1=a.d1,
        band=a.band, levels=a.levels, cross=a.cross, photo=a.photo,
        lpips=lpips, stage_fw=a.stage_fw, hyp=a.hyp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
