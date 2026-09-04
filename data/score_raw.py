"""Day 3 / 品質スコアリング — RAW モザイクから直接測る (SPEC §6.4 step 2)。

**なぜ JPEG ではなく RAW なのか**

隣接するカメラ JPEG を使えば I/O は軽いが、選抜が歪む:

  - 白飛び判定が不可能。JPEG のハイライトはトーンカーブのショルダーで潰れており、
    RAW には 1〜2段の余裕が残っていることが多い。特に Fuji の DR200/DR400 は
    RAW を意図的にアンダー露光して JPEG 側で持ち上げるため乖離が大きい。
    JPEG の見かけで落とすと **救えるデータを捨てる**。
  - シャープネスが比較不能。カメラ内シャープネスとフィルムシミュレーションが
    ショットごとに違うので、高周波エネルギーの絶対値に意味がない。
  - 暗部ノイズが測れない。カメラ内 NR が高周波を潰している。

一方、デモザイクを通さないモザイク平面なら、これらすべてが真値で測れる。
コストも安い: フル現像 2.96s/枚に対し **0.43s/枚**(実測中央値)。
支配的なのはファイルオープンで、指標計算自体は 0.03〜0.05s。

JPEG が妥当なのはバースト/近重複検出 (知覚ハッシュ) と構図統計のみ。
そちらは data/score_jpeg.py が担当する。

**全件を採点する理由**: shard rotation (§6.4) で学習期間中に別の 6,000枚を
選び直すため、品質表は一度作れば何度も使える。

**指標の改訂 (v2)**: 初版の `sharp` (lag-6 勾配の平均) はノイズとディテールを
分離できていなかった。ショットノイズは白色なのでどの lag でも同じだけ勾配に寄与し、
全32,787枚では ISO とともに増加した (ISO<=200 で 0.0416 -> ISO1601+ で 0.0506)。
このまま上位で選抜すると**ノイズの多い画像を優先**してしまい、
「Stage C 用に低ISO 高精細を集める」という §6.4 の意図と逆になる。

v2 はブロック勾配エネルギーの低位パーセンタイルを「ノイズ床」とみなして差し引く
(平坦ブロックはノイズしか含まないので σ² の推定になる):

    noise  = percentile(blocks, 5)              ノイズ床
    detail = percentile(blocks, 75) - noise     ノイズを引いた実ディテール
    snr    = detail / noise                     選抜に使う主指標

probes/p7_sharpness_metric.py による検証 (ISO帯ごと50枚):
`noise` は ISO とともに単調増加 (x3.9 = 推定器が正しく動いている)、
`detail` は減少、`snr` は 59.8 -> 9.1 と強く分離した。
副次的に、画像ごとのノイズ床推定は §6.3 のノイズ較正にもそのまま使える。

出力: cache/score_raw.jsonl。中断・再開可能。

  pixi run python data/score_raw.py
  pixi run python data/score_raw.py --limit 200

注意: /Volumes/PHOTOS 01 は読み出しのみ。一切書き込まない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import libraw_enhanced as lre

INDEX = "cache/index.jsonl"
OUT = "cache/score_raw.jsonl"

SAT = 16383          # Fuji 14bit。X-E3 / X-T4 / X-T5 とも実測で一致
XTRANS_PERIOD = 6    # X-Trans の CFA 周期。この lag なら同色サイト同士の差分になる
BLOCK = 96           # テクスチャ判定のブロック辺長 (元解像度)
CROP = 0.10          # マスク境界を避けるため上下左右を 10% 落とす


def p(*a):
    print(*a, flush=True)


def score_one(path: str) -> dict:
    """1枚を採点する。デモザイクも色変換も行わない。"""
    with lre.imread(path) as raw:
        m = raw.raw_image
        s = raw.sizes
        if m is None:
            raise RuntimeError("raw_image is None")
        a = np.asarray(m).reshape(s.raw_height, s.raw_width)

        # マスク境界 (raw_width > width) を避けるため中央を使う
        h, w = a.shape
        y0, y1 = int(h * CROP), int(h * (1 - CROP))
        x0, x1 = int(w * CROP), int(w * (1 - CROP))
        act = a[y0:y1, x0:x1]

        # 参考値。Fuji の RAF はマスク境界が狭く (X-E3 で raw_width 6048 vs
        # width 6032 = 左右合わせて 16px)、どの辺にあるかも機種依存なので、
        # ここが光学黒である保証はない。黒レベルの実用的な目安は下の p001
        # (X-E3 実測で 1064、Fuji 14bit の黒レベル ~1022 に整合する)。
        border = int(np.median(a[:, :8])) if w > 16 else 0

        sat_obs = int(act.max())
        clip = float((act >= SAT - 4).mean())
        near_clip = float((act >= SAT * 0.71).mean())   # 飽和から 1/2 段

        q = np.percentile(act[::8, ::8], [0.1, 1, 50, 99, 99.9])

        # --- シャープネス / ノイズ / テクスチャ ---
        # ::2 で間引くと X-Trans 周期 6 は lag 3 に対応する (同色サイトを維持)
        sub = act[::2, ::2].astype(np.float32)
        lag = XTRANS_PERIOD // 2
        dh = sub[:, lag:] - sub[:, :-lag]
        dv = sub[lag:, :] - sub[:-lag, :]
        mean = float(sub.mean()) or 1.0

        # 初版指標。ノイズとディテールが混ざるので選抜には使わない (下記参照)
        sharp = float(np.abs(dh).mean() / mean)

        # ブロックごとの勾配エネルギー。低位パーセンタイルが「ノイズ床」になる
        # (平坦なブロックはノイズしか含まないため σ² の推定になる)。
        g2 = dh * dh
        bs = BLOCK // 2
        gh, gw = g2.shape[0] // bs, g2.shape[1] // bs
        if gh and gw:
            blocks = np.sort(
                g2[:gh * bs, :gw * bs].reshape(gh, bs, gw, bs).mean(axis=(1, 3)).ravel())
            n = len(blocks)
            noise = float(blocks[int(n * 0.05)])
            detail = max(float(blocks[int(n * 0.75)]) - noise, 0.0)
            detail95 = max(float(blocks[int(n * 0.95)]) - noise, 0.0)
            texture = float((blocks > noise * 3.0).mean())
            n_blocks = int(gh * gw)
        else:
            noise = detail = detail95 = texture = 0.0
            n_blocks = 0

        m2 = mean ** 2
        # 手ブレは片方向にだけ勾配を潰すので、水平/垂直の非対称性が手掛かりになる
        ah, av = float(np.abs(dh).mean()), float(np.abs(dv).mean())
        aniso = abs(ah - av) / max(ah + av, 1e-6)

    return dict(
        path=path,
        sat_obs=sat_obs, border=border,
        clip=round(clip, 6), near_clip=round(near_clip, 6),
        p001=int(q[0]), p01=int(q[1]), p50=int(q[2]),
        p99=int(q[3]), p999=int(q[4]),
        sharp=round(sharp, 6),
        noise=round(noise / m2, 9), detail=round(detail / m2, 8),
        detail95=round(detail95 / m2, 8),
        snr=round(detail / max(noise, 1e-6), 4),
        texture=round(texture, 5), aniso=round(aniso, 5), n_blocks=n_blocks,
    )


def load_index(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def load_done(out: str) -> set[str]:
    if not os.path.exists(out):
        return set()
    done = set()
    with open(out) as f:
        for line in f:
            try:
                done.add(json.loads(line)["path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(a.index):
        p(f"索引がない: {a.index}  先に data/index.py を実行すること")
        return 1

    recs = load_index(a.index)
    p(f"索引 {len(recs):,} 件")
    if a.limit:
        recs = recs[:a.limit]

    done = load_done(a.out)
    todo = [r for r in recs if r["path"] not in done]
    if done:
        p(f"  処理済み {len(done):,} 件をスキップ -> 残り {len(todo):,} 件")
    if not todo:
        p("完了済み")
        return 0

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.perf_counter()
    n_err = 0
    with open(a.out, "a") as fo:
        for i, r in enumerate(todo, 1):
            try:
                rec = score_one(r["path"])
                rec.update(body=r.get("body"), shoot=r.get("shoot"),
                           iso=r.get("iso"), lens=r.get("lens"),
                           datetime=r.get("datetime"))
            except Exception as e:
                n_err += 1
                rec = dict(path=r["path"], error=f"{type(e).__name__}: {e}")
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                fo.flush()
                el = time.perf_counter() - t0
                p(f"  {i:>6,}/{len(todo):,}  {el/60:5.1f}分  "
                  f"{el/i:.2f}s/枚  残り約 {el/i*(len(todo)-i)/60:5.1f}分  "
                  f"エラー {n_err}")

    el = time.perf_counter() - t0
    p(f"\n完了: {a.out}  {el/60:.1f}分  エラー {n_err} 件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
