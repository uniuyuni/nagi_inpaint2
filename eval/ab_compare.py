"""盲検 A/B 比較ツール。

**なぜこれが要る (背景)**

このプロジェクトは指標に何度も裏切られてきた。
  - PSNR が LaMa より高い (14.1 vs 13.5 dB) のに、目視では明らかに劣る
    (白いぼけた塊が残る) ケースが複数あった。
  - 自作の簡易 LPIPS (VGG19 特徴距離、層は等重み、学習済み線形層なし) も
    同じ危険がある。
  - 人物ベンチ 60ケースでの 0.0059 対 0.0052 という差が**統計的に有意かを
    一度も確認していない**。

**効く理屈**: 最終的な品質判定を**人間の盲検比較**に置き換えれば、指標の
妥当性という不確実性が構造的に消える。加えて対比較の統計を出せば、雑音を
追いかける事故が防げる。

**効かない条件**: 提示順に偏りがあると判定が汚染される (常に右が我々、など)。
左右をケースごとに無作為化し、**判定が終わるまでどちらがどちらか分からない**
ようにすること。ラベルには手法名・PSNR などの数値を一切書かない
(先入観を排除するため)。

合成経路は `eval/quickval.py` の `quick_eval` と全く同一にしてある
(`sbs.inpaint(..., topk=8, adaptive=True, d0=4, d1=16)` ->
`stage_d.harmonize(do_color=False, do_poisson=True, do_grain=False)` ->
`np.where(mask, out, img)`)。二重実装をやめる、という同じ理屈をここでも守る。

生成モード (既定: LaMa 対 我々):
  pixi run python eval/ab_compare.py --build --spec cache/bench_person_val.json \
      --stage-f checkpoints/stage_f96/last.pt --cases 20 --out out/ab

生成モード (`--mode postproc`: LaMa 素 対 LaMa+Stage D 後処理)
  `eval/postprocess_eval.py` の docstring にある通り、LaMa の出力に Stage D を
  後処理として掛けると数値指標では良くなって見えても、盲検で見ると別の話に
  なる可能性がある (Stage F 単体への後処理適用で PSNR 24.35 -> 23.05 と悪化
  した前例がある)。この盲検A/B画像はその確認用。合成経路は
  `eval.postprocess_eval.POSTPROC_CONFIGS` を import して使う (二重実装しない)。
  `--stage-f` は使わない (StageF を一切読み込まない)。
  pixi run python eval/ab_compare.py --build --mode postproc --postproc poisson \
      --spec cache/bench_person_val.json --cases 20 --out out/ab_pp

集計モード:
  pixi run python eval/ab_compare.py --tally --out out/ab
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from eval.benchmark import load_shard_tiles   # noqa: E402
from eval.compare_visual import crop_box, to_u8  # noqa: E402
from eval.postprocess_eval import POSTPROC_CONFIGS  # noqa: E402
from masks import sample_mask                 # noqa: E402
from nagi_inpaint2 import stage_b_search as sbs  # noqa: E402
from nagi_inpaint2 import stage_d                # noqa: E402
from nagi_inpaint2.stage_f import StageF          # noqa: E402

# quick_eval / eval/benchmark.py の "クエリ+StageB+StageD" 経路と同じ既定値
# (eval/quickval.py の quick_eval のデフォルト引数)。ここでも同じ値を使うことで
# 「盲検比較で見ている出力」と「学習中/最終評価で測っている出力」を一致させる。
TOPK = 8
ADAPTIVE = True
D0 = 4.0
D1 = 16.0

VOTES_HEADER = """\
# eval/ab_compare.py --tally が読む投票ファイル。
#
# 1行1ケース、書式: <ケース名> <A|B|同等>
#   例:  X-T4_DSCF6281_018 A
#        X-T4_DSCF6281_031 同等
#
# 判定基準: 拡大切り出しを見て、写真としてどちらが自然かで選ぶこと。
# 数値的な指標は使わないこと (先入観を排除するため、ラベルには手法名や
# PSNR などの数値を書いていない)。判定が終わるまで key.json を見ないこと。
#
# 以下にケース名を1行ずつ追加してあるので、行末に A / B /同等 を書き足すこと。
"""


def p(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# 生成モード
# --------------------------------------------------------------------------

def _lama_output(lama, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f = lama(img, mask)
    return np.where(mask[..., None], f, img)


def _ours_output(sf, img: np.ndarray, mask: np.ndarray, satm: np.ndarray,
                 device: str) -> np.ndarray:
    """我々の出力を作る。合成経路は quick_eval と同一 (docstring 冒頭参照)。"""
    x = torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    mk = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
    with torch.inference_mode():
        q = torch.where(mk > 0.5, sf(x, mk), x)
        o1 = sbs.inpaint(x, mk, query=q, topk=TOPK, adaptive=ADAPTIVE,
                         d0=D0, d1=D1)[0].permute(1, 2, 0).cpu().numpy()
    d = stage_d.harmonize(o1, img, mask, valid=~satm,
                          rng=np.random.default_rng(0),
                          do_color=False, do_poisson=True, do_grain=False)
    return np.where(mask[..., None], d, img)


def _postproc_output(lama, img: np.ndarray, mask: np.ndarray, satm: np.ndarray,
                     postproc: str) -> tuple[np.ndarray, np.ndarray]:
    """`--mode postproc` 用: 「LaMa 素」と「LaMa+<postproc>」の2枚を作る。

    `postproc` は `eval.postprocess_eval.POSTPROC_CONFIGS` のキー
    (poisson/poisson_color/poisson_grain/all)。**Stage D には LaMa の生出力
    (マスク外にも伸びている) をそのまま渡す** — `eval/postprocess_eval.py`
    docstring の契約判断と同じ理由 (マスクで合成した版を渡すと調和膜が消える)。
    """
    fill = lama(img, mask)
    out_lama = np.where(mask[..., None], fill, img)
    out_post = stage_d.harmonize(fill, img, mask, valid=~satm,
                                 rng=np.random.default_rng(0),
                                 **POSTPROC_CONFIGS[postproc])
    return out_lama, out_post


def assign_sides(case_names: list[str], seed: int) -> dict[str, bool]:
    """ケースごとに「我々が A か」を無作為に決める。

    左右の無作為化が偏らないこと・seed が同じなら再現できることの両方を、
    `tests/test_ab.py` が実測で確認する。1ケースにつき1回のコイン投げに
    独立した RNG を使い、ケースの追加順に依存しない安定な割り当てにする
    (`np.random.default_rng(seed)` を1本のストリームとして順に消費する)。
    """
    rng = np.random.default_rng(seed)
    return {name: bool(rng.random() < 0.5) for name in case_names}


def build_case_image(img: np.ndarray, mask: np.ndarray, out_lama: np.ndarray,
                     out_ours: np.ndarray, ours_is_a: bool, tile: int,
                     thumb: int) -> Image.Image:
    """1ケース1枚の比較画像を作る。

    上段: 元画像 | 入力(穴)。下段: A | B (どちらが我々かはケースごとに無作為)。
    ラベルは「A」「B」のみで、手法名・数値は一切書かない (先入観の排除)。
    マスク周辺を拡大した切り出しを使う (細く小さいマスクは全体像では見えない)。
    """
    masked = img.copy()
    masked[mask] = 0.45

    y0, x0, s = crop_box(mask, tile)
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))

    a_img = out_ours if ours_is_a else out_lama
    b_img = out_lama if ours_is_a else out_ours

    gap, lab = 8, 26
    W = thumb * 2 + gap
    H = (thumb + lab) * 2
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)

    def paste(arr, col, row, text):
        im = Image.fromarray(to_u8(arr[sl])).resize((thumb, thumb), Image.NEAREST)
        x = col * (thumb + gap)
        y = row * (thumb + lab)
        sheet.paste(im, (x, y))
        dr.text((x + 4, y + thumb + 4), text, fill=(235, 235, 235))

    paste(img, 0, 0, "元画像")
    paste(masked, 1, 0, "入力(穴)")
    paste(a_img, 0, 1, "A")
    paste(b_img, 1, 1, "B")
    return sheet


def build(spec_path: str, stage_f_path: str, cases: int, out: str, device: str,
         seed: int, thumb: int = 320, mode: str = "ours",
         postproc: str = "poisson") -> None:
    """盲検A/B画像を作る。

    `mode="ours"` (既定・従来どおり): 「LaMa 素」対「我々 (StageF+コピー+
    Stage D)」。`--stage-f` の checkpoint を読み込む。

    `mode="postproc"`: 「LaMa 素」対「LaMa+<postproc>」。`eval/postprocess_eval.py`
    docstring の背景・契約判断を参照。StageF は一切読み込まない
    (`stage_f_path` は無視する)。
    """
    from eval.lama_baseline import Lama

    spec = json.load(open(spec_path))
    sel = spec["cases"][:cases]
    case_names = [f"{c['shard']}::{c['key']}" for c in sel]
    sides = assign_sides(case_names, seed)

    sf = None
    if mode == "ours":
        fck = torch.load(stage_f_path, map_location=device)
        base = int(fck.get("base") or fck["model"]["out.weight"].shape[1])
        sf = StageF(base=base).to(device)
        sf.load_state_dict(fck["model"])
        sf.eval()
        p(f"Stage F: {stage_f_path} (step {fck.get('step')}, base {base})")
    elif mode == "postproc":
        p(f"後処理モード: LaMa 素 対 LaMa+{postproc}")
    else:
        raise ValueError(f"未対応の mode: {mode!r} (ours か postproc)")
    lama = Lama(device=device)

    os.makedirs(out, exist_ok=True)
    by_shard: dict[str, list] = {}
    for c in sel:
        by_shard.setdefault(c["shard"], []).append(c)

    key_map: dict[str, str] = {}   # ケース名 -> "A" | "B" (どちらが我々/後処理側か)
    n = 0
    for shard, cs in by_shard.items():
        tiles = {k: (im, sm, mt) for k, im, sm, mt
                in load_shard_tiles(os.path.join("shards", shard), stride=1)}
        for c in cs:
            if c["key"] not in tiles:
                continue
            img, satm, _ = tiles[c["key"]]
            mask, _ = sample_mask(c["tile"], np.random.default_rng(c["mask_seed"]),
                                  kind=c["kind"])
            name = f"{c['shard']}::{c['key']}"
            alt_is_a = sides[name]

            if mode == "postproc":
                out_base, out_alt = _postproc_output(lama, img, mask, satm, postproc)
            else:
                out_base = _lama_output(lama, img, mask)
                out_alt = _ours_output(sf, img, mask, satm, device)

            sheet = build_case_image(img, mask, out_base, out_alt, alt_is_a,
                                     c["tile"], thumb)
            fname = f"{n:03d}_{c['key']}.png"
            sheet.save(os.path.join(out, fname))
            key_map[name] = "A" if alt_is_a else "B"
            n += 1
            p(f"  {n}: {name}")

    with open(os.path.join(out, "key.json"), "w") as f:
        json.dump(dict(spec=spec_path, seed=seed, cases=key_map), f,
                  indent=1, ensure_ascii=False)

    votes_path = os.path.join(out, "votes.txt")
    with open(votes_path, "w") as f:
        f.write(VOTES_HEADER)
        for name in key_map:
            f.write(f"{name} \n")

    p(f"\n-> {out} ({n} ケース)")
    p(f"   {os.path.join(out, 'key.json')} (判定が終わるまで見ないこと)")
    p(f"   {votes_path} に判定を記入すること")


# --------------------------------------------------------------------------
# 統計
# --------------------------------------------------------------------------

def binomial_test_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """両側二項検定の p 値を自前計算する (scipy 依存を増やさないため)。

    帰無仮説: 各試行が確率 p0 で成功する。観測 k 勝 n 戦がどれだけ極端かを、
    「観測した確率質量以下のすべての結果」を足し上げる標準的な定義で測る。
    p0=0.5 では二項分布が対称なので、`math.comb` で愚直に列挙しても
    n が小さい (盲検比較は多くて数十ケース) 前提では十分速い。
    """
    if n == 0:
        return float("nan")
    obs = math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
        if pi <= obs * (1 + 1e-9):
            total += pi
    return float(min(1.0, total))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval (既定 z はその臨界値)。n=0 なら (nan, nan)。"""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def tally_votes(votes: dict[str, str], key_map: dict[str, str]) -> dict:
    """`votes.txt` の判定 (A/B/同等) を `key.json` (どちらが我々か) と突き合わせる。

    戻り値: wins/losses/ties の件数、我々の勝率、95%信頼区間 (Wilson)、
    二項検定 p 値 (「同等」は除外)。
    """
    wins = losses = ties = 0
    for name, vote in votes.items():
        if name not in key_map:
            continue
        ours_side = key_map[name]
        if vote in ("同等", "=", "tie", "="):   # 記入者が使う表記の揺れを吸収
            ties += 1
        elif vote == ours_side:
            wins += 1
        elif vote in ("A", "B"):
            losses += 1
    n = wins + losses
    winrate = wins / n if n else float("nan")
    lo, hi = wilson_interval(wins, n)
    pval = binomial_test_two_sided(wins, n)
    return dict(wins=wins, losses=losses, ties=ties, n=n, winrate=winrate,
               ci_lo=lo, ci_hi=hi, pvalue=pval)


def parse_votes_file(path: str) -> dict[str, str]:
    votes: dict[str, str] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, vote = parts[0], parts[1]
        votes[name] = vote
    return votes


def tally(out: str) -> dict:
    key_map = json.load(open(os.path.join(out, "key.json")))["cases"]
    votes = parse_votes_file(os.path.join(out, "votes.txt"))
    result = tally_votes(votes, key_map)

    p(f"投票 {len(votes)} 件 / ケース {len(key_map)} 件")
    p(f"我々の勝ち {result['wins']}  負け {result['losses']}  同等 {result['ties']}")
    if result["n"]:
        p(f"勝率 {result['winrate']*100:.1f}%  "
          f"95%CI [{result['ci_lo']*100:.1f}, {result['ci_hi']*100:.1f}]%  "
          f"(n={result['n']}, 同等除く)")
        p(f"両側二項検定 p = {result['pvalue']:.5f}")
    else:
        p("勝敗の付いた投票が無い (votes.txt を確認すること)")
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--spec", default="cache/bench_person_val.json")
    ap.add_argument("--stage-f", default="checkpoints/stage_f96/last.pt")
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--out", default="out/ab")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thumb", type=int, default=320)
    ap.add_argument("--mode", choices=["ours", "postproc"], default="ours",
                    help="build の比較対象。ours=LaMa対我々 (既定・従来どおり)、"
                         "postproc=LaMa素対LaMa+後処理")
    ap.add_argument("--postproc",
                    choices=["poisson", "poisson_color", "poisson_grain", "all"],
                    default="poisson",
                    help="--mode postproc のときに使う Stage D 後処理構成")
    a = ap.parse_args()

    if a.build:
        build(a.spec, a.stage_f, a.cases, a.out, a.device, a.seed, a.thumb,
             mode=a.mode, postproc=a.postproc)
        return 0
    if a.tally:
        tally(a.out)
        return 0
    p("--build か --tally のどちらかを指定すること")
    return 1


if __name__ == "__main__":
    sys.exit(main())
