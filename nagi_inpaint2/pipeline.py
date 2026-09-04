"""L0〜L5 を通した除去パイプライン (docs/DESIGN_V2_JA.md §4)。

**これまで存在しなかったもの。** Phase 0 では 512² クロップを受け取る除去器と、
ベンチ生成に埋め込まれた切り出しコードしか無く、
「1 億画素の写真とマスクを渡したら結果が返る」実体が無かった。

    L1  マスク周りを切り出す。**大きい対象は縮小して押し込む** (L4 の入口)
    L2  自動露出でモデルの学習分布へ写し、厳密に逆写像する
    L3  除去コア (lama / yoeo / smarteraser)
    L4  縮小した場合の復元 (`detail` で方式を選ぶ)
    L5  Poisson 合成
    --  マスク外は原画素を**ビット単位で保持**

**計算量はマスク面積のみに依存する。** 画像全体の画素数には依存しないので、
1 億画素でもクロップ 1 枚分のコストで済む (SPEC §1.1 の要求)。

**入出力は知覚空間** `p = sign(x)·(|x|/white)^(1/2.2)`。
リニア ProPhoto からの変換は `data/extract.py` の担当 (上流)。

**大きい対象について (実測、§3.1.5)**

510 件中 156 件 (31%) が 512² に収まらない。その内訳:

    長辺  384〜 768px   51 件   必要な縮小 1.2 倍
    長辺  768〜1536px   58 件   必要な縮小 1.8 倍   <- ここが本番
    長辺 1536px 超      47 件   必要な縮小 5.7 倍   <- 主被写体。消す対象ではない

**縮小 2 倍で穴の中の情報量は 1/4 になる。** L4 はこれを埋める層。

**Real-ESRGAN を試したが棄却した (2026-08-25、実測)**

R17 に従い、自作の前に学習済み超解像を試した (BSD-3-Clause、16.7M params、
`strict=True` で 702 テンソル一致を確認)。**正解のある比較 (元の 512 画像を
縮小 -> 拡大) で bilinear に負けた**:

| 縮小率 | 手法 | 知覚距離↓ | PSNR↑ | 高周波 (100% が目標) |
|---|---|---|---|---|
| 2倍 | bilinear | **0.00126** | **40.90** | 48% |
| 2倍 | Real-ESRGAN | 0.00289 | 30.89 | **172%** |
| 4倍 | bilinear | **0.00286** | **35.37** | 35% |
| 4倍 | Real-ESRGAN | 0.00433 | 26.49 | **208%** |

Real-ESRGAN は**ぼけ・ノイズ・JPEG という「実世界の劣化」**を戻すよう学習されている。
こちらの入力は高品質な写真をきれいに縮小しただけなので、**過剰に detail を足す**。
継ぎ目が 1.05 -> 1.15 に悪化したのも、穴の中だけ周囲より鮮鋭になるため。

**用途が違う。** 穴の中には復元すべき正解が存在しないので、
目標は「detail の最大化」ではなく「**周囲との整合**」である。
HF 移植は周囲の高周波を持ってくるので、定義上そこが揃う。
既定は `detail="hf"`。

**ただし L4 の経路そのものが高くつく (2026-08-30、盲検A/Bで実測)**

同一対象を「512原寸で処理」対「1024を512に縮小して処理し戻す」で比べたところ、
**512原寸の 12 勝 1 敗** (n=13、両側二項検定 p=0.0034、`eval/ab_context.py`)。
自動指標 (継ぎ目 1.07 同点 / grain 0.849 対 0.852 / 人物残存 7 対 4) は
**逆に縮小側を推していた**。指標と人手の不一致はこれで 6 度目。

負け方は「細部が甘い」ではなく **構造ごと潰れる**:
欄干の桟が灰色の板になり、煉瓦の目地が滑らかな諧調に溶ける。
**HF 移植は統計 (grain) を戻すだけで構造は戻せない。**
grain が揃っていることは「正しいものが描かれている」ことを意味しない。

したがって `crop=512` は動かさない (視野を広げる方向は閉じた)。
512² に収まらない 31% の対象はこの経路を通らざるを得ないが、
**そこは 12対1 で負ける品質だと自覚して扱うこと。**

**なぜ大きいクロップのまま処理しないのか**: M1 の実測で
768² は 29.5 秒 (512² の 5 倍)、1024² は fp32/fp16 ともメモリで不可
(潜在 128² = 16,384 トークンの注意行列が 8.6GB で MPS の上限超え)。
**512² が唯一の実用点**なので、大きい対象は縮小するしかない。
"""
from __future__ import annotations

import numpy as np

from nagi_inpaint2.removers import build as build_remover
from nagi_inpaint2.removers import composite

DETAIL_MODES = ("none", "hf", "esrgan")


def _resize(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """bilinear 縮小拡大 (torch を使う。numpy だけだと品質が足りない)。"""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x)).float()
    if t.ndim == 2:
        t = t[None, None]
        out = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
        return out[0, 0].numpy()
    t = t.permute(2, 0, 1)[None]
    out = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return out[0].permute(1, 2, 0).numpy()


def _highpass(x: np.ndarray, k: int = 3) -> np.ndarray:
    """箱平均を引いた高周波帯 (eval/metrics.py の `_highpass` と同じ流儀)。"""
    import scipy.ndimage as ndi
    if x.ndim == 2:
        return x - ndi.uniform_filter(x, k)
    return np.stack([x[..., c] - ndi.uniform_filter(x[..., c], k)
                     for c in range(x.shape[-1])], -1)


def inject_hf(filled: np.ndarray, src: np.ndarray, mask: np.ndarray,
              target: float = 0.95) -> np.ndarray:
    """縮小で失われた高周波を、**既知領域から鏡像反射して**穴へ移植する。

    縮小 2 倍で穴の中の情報量は 1/4 になり、穴だけツルツルになる
    (実測: grain 0.781 -> 0.622)。既知画素の高周波を持ち込んで埋める。

    **初版のバグ (2026-08-25)**: `hf[mask]` を足していた。
    これは**消したい物体そのものの高周波**であり、
    消した跡に物体のテクスチャを戻し入れていた。
    grain は 0.718 まで回復したのに参照との知覚距離が
    0.00465 -> 0.00468 と全く改善しなかったのはこのため。
    **grain だけ見ていたら気づけなかった。**

    正しくは、穴の各画素について**マスク境界をまたいで鏡像の位置**にある
    既知画素の高周波を採る。最近傍をそのまま複製すると筋になるので、
    `p_src = 2 * p_nearest - p_hole` で反射させる。
    """
    import scipy.ndimage as ndi
    m = mask.astype(bool)
    known = ~m
    if known.sum() < 256 or m.sum() < 16:
        return filled

    hf_src = _highpass(src)
    s_known = float(hf_src[known].std())
    s_hole = float(_highpass(filled)[m].std())
    if s_known < 1e-6 or s_hole >= s_known * target:
        return filled

    # 穴の各画素 -> 最も近い既知画素の座標
    _, idx = ndi.distance_transform_edt(m, return_indices=True)
    yy, xx = np.nonzero(m)
    ny, nx = idx[0][m], idx[1][m]
    # 境界をまたいで鏡像反射した位置 (最近傍の複製だと筋になる)
    ry = np.clip(2 * ny - yy, 0, m.shape[0] - 1)
    rx = np.clip(2 * nx - xx, 0, m.shape[1] - 1)
    # 反射先がまだ穴の中なら最近傍で妥協する
    inside = m[ry, rx]
    ry[inside], rx[inside] = ny[inside], nx[inside]

    need = np.sqrt(max(s_known ** 2 - s_hole ** 2, 0.0)) / max(s_known, 1e-6)
    out = filled.copy()
    out[yy, xx] = filled[yy, xx] + hf_src[ry, rx] * need
    return out


def crop_box(mask: np.ndarray, crop: int, shape: tuple[int, int],
             margin: float = 0.25) -> tuple[int, int, int, int, int]:
    """マスクを含む正方形の切り出し範囲を決める。

    戻り値 `(y0, x0, side, scale_num, scale_den)`。
    `side` は原寸での一辺で、`crop` より大きければ縮小して押し込む。

    対象の外接矩形に `margin` の余白を足した正方形を取る。
    余白は「モデルに文脈を見せる」ためで、
    §3.1.5 の被覆判定でも 25% を見込んでいる。
    """
    H, W = shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0, 0, min(crop, H, W), 1, 1
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    need = int(round(max(bh, bw) * (1.0 + 2 * margin)))
    side = max(crop, need)
    side = min(side, H, W)                 # 画像からはみ出さない
    cy = int((ys.min() + ys.max()) // 2)
    cx = int((xs.min() + xs.max()) // 2)
    y0 = int(np.clip(cy - side // 2, 0, H - side))
    x0 = int(np.clip(cx - side // 2, 0, W - side))
    return y0, x0, side, side, crop


class MultiInpainter:
    """**候補を複数返す**除去器。`multi(img, mask) -> {手法名: 画像}`。

    **なぜ 1 案ではないのか (2026-08-30、ユーザ判断)**

    ShadeWave は写真アプリなので、結果を見せて**採否をユーザに委ねられる**。
    そこには精度 100% の判定者が既に居るので、L6 検品 (機械で採否を決める層) は
    作らないことにした。**その決定が、返すものを 1 案から複数案に変える。**

    **実測 (実除去ベンチ 60 件、絶対評価。`eval/accept.py`)**

    | | 採用率 | 95%CI | 秒/件 |
    |---|---|---|---|
    | LaMa 単独 | 63.3% | [50.7%, 74.4%] | 0.63 |
    | YOEO+L5 単独 | 75.0% | [62.8%, 84.2%] | 5.3 |
    | **2 案** | **88.3%** | **[77.8%, 94.2%]** | **5.9** |

    **上積み +13.3 点、費用 +0.63 秒。** LaMa だけ採用可だったケースが 8 件あり、
    **LaMa は YOEO の劣化版ではない** (得意な場面が違う)。
    McNemar p=0.21 なので「YOEO が上」も統計的には言えていない。
    どちらか一方を選ぶ根拠が無い以上、**両方出すのが正しい**。

    **クロップは 1 回しか取らない。** 切り出し・縮小はコア間で共有し、
    コアの数だけ増えるのは推論だけである。
    """

    DEFAULT = ("lama", "yoeo")

    def __init__(self, cores: tuple[str, ...] = DEFAULT, device: str = "mps",
                 crop: int = 512, detail: str = "hf", poisson: bool = True,
                 **core_kw):
        if not cores:
            raise ValueError("cores が空")
        self.cores = tuple(cores)
        # YOEO は 2 step が実測の最良 (3/4 step は改善せず)。
        kw = {"yoeo": dict(steps=2)}
        self.inp = {c: Inpainter(core=c, device=device, crop=crop,
                                 detail=detail, poisson=poisson,
                                 **{**kw.get(c, {}), **core_kw})
                    for c in self.cores}
        self.crop = crop

    # ------------------------------------------------------------------
    def patches(self, img: np.ndarray, mask: np.ndarray):
        """`(y0, x0, {手法名: パッチ})` を返す。**1 億画素では必ずこちら。**

        `__call__` は画像全体を手法の数だけ複製するので、
        100MP・2 案なら複製だけで 2.4GB になる。
        """
        mask = np.asarray(mask, bool)
        H, W = img.shape[:2]
        y0, x0, side, _, _ = crop_box(mask, self.crop, (H, W))
        # **切り出してから型変換する。** 先に画像全体を float32 で複製すると
        # 100MP で 1.26GB の複製が入り、「費用は画素数に依存しない」が壊れる
        # (実測 4.2MP 7.95 秒に対し 100MP 13.82 秒。2026-08-30 に検出)。
        sub = np.ascontiguousarray(img[y0:y0 + side, x0:x0 + side]
                                   .astype(np.float32))
        sm = np.ascontiguousarray(mask[y0:y0 + side, x0:x0 + side])
        if sm.sum() == 0:
            return y0, x0, {c: sub.copy() for c in self.cores}

        out = {}
        for c in self.cores:
            inp = self.inp[c]
            merged = composite(inp.fill_crop(sub, sm), sub, sm,
                               poisson=inp.poisson)
            # **マスク外はビット単位で原画素**。Poisson も含め一切触らせない。
            merged[~sm] = sub[~sm]
            out[c] = merged
        return y0, x0, out

    # ------------------------------------------------------------------
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> dict:
        """全体画像を手法の数だけ返す。**大きい画像では `patches` を使うこと。**"""
        mask = np.asarray(mask, bool)
        if mask.sum() == 0:
            base = np.ascontiguousarray(img.astype(np.float32))
            return {c: base.copy() for c in self.cores}
        y0, x0, pats = self.patches(img, mask)
        res = {}
        for c, p in pats.items():
            o = np.ascontiguousarray(img.astype(np.float32))
            o[y0:y0 + p.shape[0], x0:x0 + p.shape[1]] = p
            res[c] = o
        return res


class Inpainter:
    """除去パイプライン。`inp(img, mask) -> img`。知覚空間 (H,W,3) float32。

    `core`   : "lama" / "yoeo" / "smarteraser" / "rorem"
    `crop`   : モデルに渡す一辺 (M1 の実用点は 512)
    `detail` : 縮小したときの復元方式
               "none"   — bilinear で戻すだけ (基準線)
               "hf"     — 周囲の高周波を穴へ移植する (自作)
               "esrgan" — Real-ESRGAN x2 で拡大する。**実測で棄却** (下記)
    `poisson`: L5 の Poisson 合成 (既定 on)
    """

    def __init__(self, core: str = "yoeo", device: str = "mps",
                 crop: int = 512, detail: str = "hf", poisson: bool = True,
                 **core_kw):
        if detail not in DETAIL_MODES:
            raise ValueError(f"detail は {DETAIL_MODES} のいずれか: {detail!r}")
        self.core = build_remover(core, device=device, **core_kw)
        self.core_name = core
        self.crop, self.detail, self.poisson = crop, detail, poisson
        self._up = None
        self._device = device

    # ------------------------------------------------------------------
    def fill_crop(self, sub: np.ndarray, sm: np.ndarray) -> np.ndarray:
        """クロップ 1 枚を埋める。必要なら縮小して押し込み、L4 で戻す。"""
        side = sub.shape[0]
        if side <= self.crop:
            return self.core(sub, sm)

        # --- 縮小してモデルへ (L4 の入口) ---
        small = _resize(sub, (self.crop, self.crop))
        sm_small = _resize(sm.astype(np.float32), (self.crop, self.crop)) > 0.5
        if sm_small.sum() < 16:            # 縮小でマスクが消えた
            sm_small = _resize(sm.astype(np.float32),
                               (self.crop, self.crop)) > 0.05
        filled_small = self.core(small, sm_small)

        if self.detail == "esrgan":
            # **学習済みの超解像で戻す。** bilinear は高周波を 39% しか復元しないが
            # Real-ESRGAN は 117% (実測、元画像を 100% として)。
            # Real-ESRGAN は 8bit sRGB で学習されているので [0,1] へ正規化して渡す。
            if self._up is None:
                from nagi_inpaint2.upscale import RealESRGAN
                self._up = RealESRGAN(device=self._device)
            lo = float(np.percentile(filled_small, 0.5))
            hi = float(np.percentile(filled_small, 99.5))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-4:
                lo, hi = 0.0, 1.0
            norm = np.clip((filled_small - lo) / (hi - lo), 0.0, 1.0)
            cur = self.crop
            while cur < side:                       # 2 倍ずつ上げる
                norm = self._up(norm)
                cur *= 2
            filled = norm * (hi - lo) + lo
            if filled.shape[0] != side:             # 端数は bilinear で合わせる
                filled = _resize(filled, (side, side))
            return filled.astype(np.float32)

        filled = _resize(filled_small, (side, side))
        if self.detail == "hf":
            filled = inject_hf(filled, sub, sm)
        return filled

    # ------------------------------------------------------------------
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """`img` (H,W,3) 知覚空間、`mask` (H,W) bool。マスク外はビット保持。"""
        mask = np.asarray(mask, bool)
        if mask.sum() == 0:
            return np.ascontiguousarray(img.astype(np.float32))
        y0, x0, patch = self.patch(img, mask)
        out = img.copy()
        out[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch
        return out

    # ------------------------------------------------------------------
    def patch(self, img: np.ndarray, mask: np.ndarray):
        """`(y0, x0, patch)` を返す。**1 億画素では必ずこちらを使う。**

        `__call__` は画像全体を複製するので、100MP では複製だけで十数秒かかる
        (9.4MP で 1.4 秒を実測)。呼び出し側がその場で貼れるよう、
        変更した矩形だけを返す口を用意する。
        """
        mask = np.asarray(mask, bool)
        H, W = img.shape[:2]
        y0, x0, side, _, _ = crop_box(mask, self.crop, (H, W))
        # **切り出してから型変換する** (MultiInpainter.patches と同じ理由)
        sub = np.ascontiguousarray(img[y0:y0 + side, x0:x0 + side]
                                   .astype(np.float32))
        sm = np.ascontiguousarray(mask[y0:y0 + side, x0:x0 + side])
        if sm.sum() == 0:
            return y0, x0, sub.copy()

        filled = self.fill_crop(sub, sm)
        merged = composite(filled, sub, sm, poisson=self.poisson)
        # **マスク外はビット単位で原画素**。Poisson も含め一切触らせない。
        # クロップ内だけで済むので、画像全体を走査しない。
        merged[~sm] = sub[~sm]
        return y0, x0, merged
