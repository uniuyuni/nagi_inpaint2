"""Stage B — ブロック単位の全探索による非パラメトリック合成 (SPEC §4.2)。

**多段ピラミッド実装 (`stage_b.py` / `stage_b_simple.py`) を置き換える。**
実測で、単純な全探索のほうが 7.5 dB 良かった:

    ブロック全探索 (学習なし)   23.9 dB
    多段ピラミッド + top-k       16.4 dB

多段化は「粗いレベルで大域探索し細レベルで精緻化する」という筋の良い設計に見えたが、
実装では各段でオフセットの平滑化・伝播・丸めが必要になり、そのたびに情報を失っていた。
探索範囲が 512² タイルに収まる限り、全探索のほうが速くも正確でもある。

**設計の天井（実測、高周波テクスチャ × blob）**

    ブロック  神託(正解を見る)  実測(照合コスト)   差
      64px      32.11 dB        23.87 dB      8.24
      32px      33.27 dB        23.88 dB      9.40
      16px      34.64 dB        23.89 dB     10.74

    LaMa 29.53 dB

**律速は解像度ではなく照合コストの質である。** ブロックを細かくすると神託の上限は
上がるが、正解なしで正しいオフセットを選ぶ能力が追いつかず、実測値は動かない
(23.87 → 23.89 dB)。この 8〜12 dB の差こそが学習で埋めるべきもの。

したがってブロックは 32〜64px で十分。細かくする価値は照合が良くなってから出る。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .stage_b_simple import _box, _features, _shift

SEARCH_RADIUS = 96
SEARCH_STRIDE = 4
BLOCK = 32
PATCH = 7
MIN_W = 0.05          # ブロックのコストを信用する最低限の既知画素割合


def _offsets(radius: int, stride: int):
    return [(oy, ox) for oy in range(-radius, radius + 1, stride)
            for ox in range(-radius, radius + 1, stride) if (oy, ox) != (0, 0)]


def _fill_blocks_inward(idx: torch.Tensor, valid: torch.Tensor,
                        iters: int = 64) -> torch.Tensor:
    """コストが信用できないブロックへ、周囲の決まったブロックから伝播させる。

    穴の奥のブロックは周囲が全て穴なので照合できない。近傍から埋める。
    """
    idx = idx.float()
    k = valid.float()
    for _ in range(iters):
        if float((1 - k).sum()) == 0:
            break
        nb = F.avg_pool2d(F.pad(k, (1, 1, 1, 1)), 3, stride=1)
        grow = ((nb > 1e-6).float() - k).clamp(0, 1)
        if float(grow.sum()) == 0:
            break
        # 平均ではなく最頻に近い値を採るため、近傍の中央値相当として max_pool を使う
        prop = F.max_pool2d(F.pad(idx * k, (1, 1, 1, 1)), 3, stride=1)
        idx = idx * (1 - grow) + prop * grow
        k = (k + grow).clamp(0, 1)
    return idx.long()


# 特徴を正規化する目標 RMS。手作り特徴の実測値 (~0.33) に合わせてあり、
# この値そのものに意味は無い (コストの順位は f の定数倍で不変)。重要なのは
# **どの特徴抽出器を使っても cost の絶対スケールが揃う**こと。
FEAT_RMS = 0.33


def solve_blocks(img: torch.Tensor, mask: torch.Tensor, feat=None,
                 radius: int = SEARCH_RADIUS, stride: int = SEARCH_STRIDE,
                 block: int = BLOCK, patch: int = PATCH, query=None):
    """ブロックごとに最良オフセットを全探索で決める。

    img   : (1,3,H,W) 知覚空間
    mask  : (1,1,H,W) 1 = 穴
    feat  : 特徴抽出器。None なら手作り特徴 (生RGB + 勾配)
    query : **Stage A の低周波補完 L で穴を埋めた画像** (1,3,H,W)。

    **query を与えないと穴の内部は照合できない。** 内部画素はパッチが丸ごと穴に
    覆われるため、照合可能なブロックは 36% しかなく、残りは縁からの伝播で決まる
    ("選ばれて"いない)。低周波で埋めれば内部にもクエリが生まれ、
    照合可能なブロックが 100% になる。実測:

        クエリなし              24.03 dB   (照合可能 36%)
        クエリあり (LaMa低周波)   32.75 dB   (照合可能 100%)
        神託 (天井)             33.27 dB
        LaMa                   29.53 dB

    **+8.7 dB。学習は一切していない。** 照合コストの学習が 4,000 step 効かなかったのは
    損失の問題ではなく、学習しても届かない場所が 64% あったためだった。
    Stage A は「あれば良い」ではなく、**無ければ Stage B が動かない**。

    **クエリ側とソース側で有効マスクを分ける。**
      クエリ側: L で埋めた画素も「中身がある」ので有効
      ソース側: 穴の中からコピーしてはならないので無効のまま

    戻り値: (ブロックごとのオフセット index, オフセット一覧, ブロックごとのコスト)
    """
    src = 1.0 - mask                      # コピー元として使ってよいか
    if query is None:
        qimg, qval = img, src
    else:
        qimg, qval = query, torch.ones_like(mask)
    f = (_features(qimg) if feat is None
         else feat(qimg, mask if query is None else torch.zeros_like(mask))) * qval
    # **特徴のスケールを固定する。** 下の BIG は「無効オフセットを必ず負けさせる」
    # ための番兵だが、これは絶対値の定数なので、特徴の大きさが変わると意味を失う。
    # 実測: 学習した特徴は |f| 平均 4153 (手作りは 0.33) まで膨れ、コストが BIG を
    # 追い越した結果、**穴の中からコピーするオフセットが最小コストとして選ばれた**
    # (val 16.47 -> 14.37 dB)。順位学習の損失はブロック平均で正規化されており
    # スケール不変なので、学習側には絶対値を縛る力が無い。
    # コストの順位は f の定数倍で不変なので、ここで正規化しても選択結果は変わらず、
    # 番兵だけが正しく機能するようになる。
    f = f * (FEAT_RMS / (f.pow(2).mean().sqrt() + 1e-12))
    r = patch // 2
    offs = _offsets(radius, stride)

    # **無効画素は「除外」ではなく「大きなコスト」として扱う。**
    # 除外して有効分だけ平均すると、ブロックのごく一部でしか有効でないオフセットが
    # 薄い根拠で勝ててしまう。実測で 23.9 dB -> 19.3 dB と 4.6 dB 落ちた。
    # 大きな値を入れれば「ブロック全体にわたって有効」なオフセットが選ばれる。
    BIG = 1e6
    costs = []
    for oy, ox in offs:
        vs = _shift(src, oy, ox)
        w = qval * vs
        wb = _box(w, r)
        c = _box(((f - _shift(f, oy, ox)) ** 2).sum(1, keepdim=True) * w, r) / (wb + 1e-6)
        bad = (vs <= 0.5) | (wb < MIN_W)
        c = torch.where(bad, torch.full_like(c, BIG), c)
        costs.append(F.avg_pool2d(c * mask, block))

    Cn = torch.cat(costs, 1)                   # (1,K,h,w)
    idx = Cn.argmin(1, keepdim=True)
    # 全オフセットが無効だったブロック (穴の奥) は近傍から埋める
    ok = Cn.min(1, keepdim=True).values < BIG * 0.5
    idx = _fill_blocks_inward(idx, ok)
    return idx, offs, Cn


def coherent_idx(Cn: torch.Tensor, offs, idx: torch.Tensor,
                 lam: float = 0.5, tol: float = 8.0, iters: int = 4) -> torch.Tensor:
    """`solve_blocks` の独立 argmin を近傍と整合させながら選び直す (PatchMatch の伝播に相当)。

    **平均化ではない。** `paste_topk` のように複数候補を混ぜてテクスチャをぼかす
    のではなく、あくまで「1ブロック1候補のコピー」を保ったまま、どの候補を選ぶかを
    近傍と揃える方向に倒す。regret (選んだ候補の真の誤差 / 全候補中の最良) が 1.73
    と大きいのは、ブロックが完全に独立に argmin を取るせいで、隣接ブロックが無関係な
    場所からコピーして構造が寸断されるため。実画像では隣接ブロックの正解オフセットは
    近いことが多いという事前知識を、コストに軽いペナルティとして足すことで使う。

    候補は各ブロックにつき5つ (自分の現在の `idx` + 上下左右の隣接ブロックの現在の
    `idx`) に絞る。全候補 (K=2400) を毎回舐め直すと近傍整合の意味が薄れる上に
    コストも爆発するため、「隣に伝染する」候補だけを見る PatchMatch のジャンプフラッド
    に近い設計にしている。

    `score(o) = Cn.gather(1, o) + lam * s * penalty(o)`。`s` はコストのスケール
    正規化 (`block_confidence` / `topk_weights` と同じ「有効ブロックの cmin の中央値」
    方式)。特徴抽出器を差し替えてもコストの絶対スケールが変わるだけで `lam` の意味が
    変わらないようにするため。`penalty` は候補のオフセットが4近傍の現在のオフセット
    からどれだけ画素距離で離れているか (`tol` で正規化して 0〜1 にクリップ) の4近傍平均。

    穴の奥で全候補が無効 (`cmin >= 5e5`) なブロックは、`solve_blocks` が
    `_fill_blocks_inward` で近傍伝播により決めた値をそのまま保持し、ここでは
    書き換えない (そのブロックには元々信用できるコストが無いので、コストに基づく
    再選択をしても意味がない)。
    """
    device, dtype = Cn.device, Cn.dtype
    offs_t = torch.tensor(offs, dtype=dtype, device=device)          # (K,2)

    cmin = Cn.min(dim=1, keepdim=True).values                        # (1,1,h,w)
    matched = cmin < 5e5
    vals = cmin[matched]
    if vals.numel() == 0:
        s = 1.0
    else:
        s = float(vals.median())
        if s <= 0:
            s = 1.0
    dead = ~matched                                                   # (1,1,h,w) 更新禁止

    idx = idx.clone()
    idx_dead = idx.clone()   # 無効ブロックはこの初期値のまま固定する
    _, _, h, w = idx.shape

    for _ in range(iters):
        padded = F.pad(idx.float(), (1, 1, 1, 1), mode="replicate")
        up    = padded[:, :, 0:h,     1:w + 1].round().long()
        down  = padded[:, :, 2:h + 2, 1:w + 1].round().long()
        left  = padded[:, :, 1:h + 1, 0:w    ].round().long()
        right = padded[:, :, 1:h + 1, 2:w + 2].round().long()

        cand = torch.cat([idx, up, down, left, right], dim=1)        # (1,5,h,w)
        cost = Cn.gather(1, cand)                                    # (1,5,h,w)

        cand_xy = offs_t[cand]                                       # (1,5,h,w,2)
        nb = torch.cat([up, down, left, right], dim=1)               # (1,4,h,w)
        nb_xy = offs_t[nb]                                           # (1,4,h,w,2)

        d = (cand_xy.unsqueeze(2) - nb_xy.unsqueeze(1)).norm(dim=-1)  # (1,5,4,h,w)
        penalty = torch.clamp(d / tol, max=1.0).mean(dim=2)           # (1,5,h,w)

        # **コストはブロック自身の最良コストで相対化する。**
        # 当初は画像全体の cmin 中央値 `s` で正規化していたが、5候補間のコスト差は
        # それより桁違いに小さく、ペナルティが常に支配してオフセット場が一様に
        # 潰れていた (val 16.47 -> 14.91 dB)。「最良より何割悪いか」で測れば
        # lam は「何割の悪化までなら近傍との一致と引き換えに許すか」を意味する。
        c0 = cost.min(dim=1, keepdim=True).values
        score = cost / (c0 + 1e-6 * s) + lam * penalty
        best = score.argmin(dim=1, keepdim=True)                     # (1,1,h,w)
        new_idx = cand.gather(1, best)                                # (1,1,h,w)

        idx = torch.where(dead, idx_dead, new_idx)

    return idx


def paste(img: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, offs) -> torch.Tensor:
    """ブロックごとのオフセットで貼り直す。

    **マスク外にも合成結果を残す** (§4.4 の Stage D の契約)。
    マスク外を元画像に戻すのは呼び出し側の責任。
    """
    jf = F.interpolate(idx.float(), size=img.shape[-2:], mode="nearest").long()[0, 0]
    out = img.clone()
    for i in torch.unique(jf).tolist():
        oy, ox = offs[int(i)]
        m = (jf == i)[None, None].expand_as(img)
        out = torch.where(m, _shift(img, oy, ox), out)
    return out


def paste_feather(img: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, offs,
                  block: int = BLOCK, overlap: float = 0.5) -> torch.Tensor:
    """`paste` の重なりあり + ハン窓フェザリング版。ブロック境界の継ぎ目を消す。

    **なぜ必要か**: `paste` は `block` 格子を隙間なく敷き詰めて貼るだけなので、
    隣接ブロックが無関係な場所からコピーすると格子の境目がそのまま画素の不連続
    として残る (目視検証で確認済み。幅のあるマスクほど目立つ — 細いマスクは
    周囲に手がかりがあるため起きにくいが、幅のあるマスクでは内部で隣接ブロックが
    無関係な場所からコピーするため 32px の正方形パッチワークが肉眼で見える)。
    古典的な処方は「ブロックを半分ずつ重ねて貼り、重み付きで平均する」テクスチャ
    合成の手法で、境界を滑らかに混ぜて不連続を消す。

    **位相 (phase)**: `overlap` の割合だけブロック格子をずらした複数のコピーを
    ハン窓 (`0.5 - 0.5*cos(2*pi*t)`) で重み付けして平均する。`overlap=0.5` なら
    格子を (0,0)/(block/2,0)/(0,block/2)/(block/2,block/2) の4通りにずらした
    位相を使う (ある位相の境界は、別の位相の窓の中央付近でちょうど覆われる)。
    `overlap<=0` なら位相は (0,0) のみで、以下で示すとおり `paste` と厳密に等価になる。

    **位相ごとにオフセットを選び直す**: `idx` はブロック解像度なので、そのまま
    画素解像度へ最近傍拡大 (`idx_px`) しただけでは `paste` と同じ格子でしか
    見えない。位相ごとに「そのブロック格子の中心にあたる画素」の `idx_px` の値を
    代表として引き直すことで、位相ごとに異なる代表オフセットを持ちうるようになる。
    これが無いと窓を掛けても同じ画素源を薄めるだけで、境界の不連続そのものは
    消えない (窓の重なりが実際に異なる画素源を混ぜて初めて継ぎ目が消える)。

    **窓を画素中心でサンプリングする**: ブロック内座標 `t=(local+0.5)/block` を
    ハン窓に渡す。`t` がちょうど 0 や 1 になる (窓の値が 0 になる) 画素が出ない
    ようにするため。ただし `overlap<=0` (位相が1つだけ) のときは、窓の値が
    ブロックの隅で 0 に近づくと `+1e-8` との比が無視できなくなり、割り算を経由
    すると `paste` からわずかにズレてしまう (窓は本来分子分母で相殺されるはずが、
    浮動小数の割り算では厳密には相殺しない)。位相が1つしかない場合はそもそも
    窓をかけて重ねる相手が無いので、この場合だけは `paste` をそのまま返し
    (下記参照)、`overlap=0.0` が `paste` と厳密に一致することを保証する。

    契約は `paste` と同じ: 戻り値は `img` と同じ shape で、**マスク外にも合成結果が
    伸びる** (マスク外を元画像に戻すのは呼び出し側の責任, §4.4)。

    ベクトル化: ブロックの二重ループは行わない。位相ごとに `torch.unique` で
    実際に出現する代表オフセットだけを回す (`paste` と同じ手法)。行と列の窓・
    代表値抽出はそれぞれ分離可能 (`_box`/`_gaussian_blur` と同じ考え方) なので、
    1次元の演算を外積するだけで済む。

    img, mask, idx, offs : `paste` と同じ
    block   : ブロックサイズ (`solve_blocks` と揃える)
    overlap : ブロック格子をずらす量 (block に対する割合)。0.5 で半分重ね、
              0 以下で `paste` と等価
    """
    device, dtype = img.device, img.dtype
    H, W = img.shape[-2], img.shape[-1]
    idx_px = F.interpolate(idx.float(), size=(H, W), mode="nearest").long()[0, 0]

    shift = int(round(block * overlap))
    if shift <= 0:
        # 位相が (0,0) だけなら、ハン窓は分子分母の両方に同じ値がかかって数学的には
        # 相殺し `paste` と同じ結果になるはずだが、窓はブロックの隅で 0 に近づくため
        # (`+1e-8` との比が無視できなくなる) 浮動小数の割り算を経由すると `paste` から
        # わずかにズレる。位相が1つしかない場合は窓をかける意味自体が無い
        # (重ねる相手が無い) ので、素の `paste` をそのまま返して厳密な一致を保証する。
        return paste(img, mask, idx, offs)

    shift = max(1, min(shift, block - 1))
    phases = [(dy, dx) for dy in (0, shift) for dx in (0, shift)]

    pi = float(np.pi)

    def _axis(n: int, d: int):
        """1軸ぶんの (ハン窓, ブロック中心座標) を返す (行・列で共用する分離可能な核)。"""
        pix = torch.arange(n, device=device)
        shifted = pix - d
        by = torch.div(shifted, block, rounding_mode="floor")
        local = shifted - by * block                       # 常に [0, block)
        t = (local.to(dtype) + 0.5) / block                 # 画素中心サンプリング
        w = 0.5 - 0.5 * torch.cos(2 * pi * t)
        center = (by * block + d + block // 2).clamp(0, n - 1)
        return w, center.long()

    numer = torch.zeros_like(img)
    denom = torch.zeros((1, 1, H, W), device=device, dtype=dtype)

    for dy, dx in phases:
        wy, cy = _axis(H, dy)
        wx, cx = _axis(W, dx)
        window = (wy[:, None] * wx[None, :])[None, None]          # (1,1,H,W)
        rep = idx_px.index_select(0, cy).index_select(1, cx)      # (H,W) 代表 idx

        out_phase = torch.zeros_like(img)
        for i in torch.unique(rep).tolist():
            oy, ox = offs[int(i)]
            m = (rep == i)[None, None].expand_as(img)
            out_phase = torch.where(m, _shift(img, oy, ox), out_phase)

        numer = numer + out_phase * window
        denom = denom + window

    return numer / (denom + 1e-8)


def block_confidence(Cn: torch.Tensor, tau_c: float = 1.0) -> torch.Tensor:
    """ブロックごとの照合信頼度 (SPEC §4.2 `C`)。

    `solve_blocks` の全候補コスト `Cn` の最小値だけを見る。**最小値がそもそも大きい
    ＝画像内に似た領域が存在しない**ブロックであり、無理にコピーすると誤った画素を
    高信頼であるかのように貼ってしまう。逆に最小値が小さいブロックは真に良い一致が
    見つかっている。

    正規化の基準 `s` は「一致が見つかったブロックの中央値コスト」を使う。画像ごと・
    テクスチャごとにコストの絶対スケールが変わるため、固定閾値では
    (a) 高周波テクスチャで全ブロックが「低信頼」判定される、
    (b) 平坦な画像で全ブロックが「高信頼」判定される、のどちらかに倒れやすい。
    中央値を基準にすれば「その画像の中で相対的にどれだけ良い一致か」を測れる。

    有効な一致が一つも無い（穴が大きすぎて全ブロックが BIG）場合や、中央値が
    0 以下（数値的に縮退している）場合は、信頼度を定義できないので全ゼロを返す
    ＝ 全ブロックを Stage A の低周波補完へ退避させる安全側の挙動になる。
    """
    cmin = Cn.min(dim=1, keepdim=True).values           # (1,1,h,w)
    matched = cmin < 5e5
    vals = cmin[matched]
    if vals.numel() == 0:
        return torch.zeros_like(cmin)
    s = vals.median()
    if float(s) <= 0:
        return torch.zeros_like(cmin)
    conf = torch.exp(-cmin / (tau_c * s)) * matched.to(cmin.dtype)
    return conf


def composite(img: torch.Tensor, mask: torch.Tensor, query: torch.Tensor,
             idx: torch.Tensor, offs, conf: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """`paste` の結果と `query` (Stage A の低周波補完) を信頼度でブレンドする。

    **信頼度ゲート合成 (SPEC §4.3 Stage C の最小実装に相当)**。良い参照元が無い
    ブロックでも `paste` は必ず「一番マシだったオフセット」をコピーしてしまい、
    それが実際には無関係な画素だと PSNR をむしろ悪化させる。信頼度が低いブロックほど
    `query`（ぼけているが構造は正しい低周波補完）側に滑らかに退避させることで、
    「悪いコピー」より「無難なぼけ」を選ぶ。学習済み Stage C は将来この退避先に
    残差テクスチャを足すが、ここではブレンドのみの最小実装。

    conf はブロック解像度 (1,1,h,w) なので、画素解像度へ最近傍で拡大してから使う。
    ブロック境界をまたいで confidence が滑らかに変わるわけではないが、ブロックの
    粒度で急に色が変わるよりは query 側への連続的な移行のほうが継ぎ目が目立たない。

    戻り値の契約は `paste` と同じ: マスク外にも合成が伸びる。マスク外を元に戻すのは
    呼び出し側の責任 (§4.4 Stage D の契約)。
    """
    copy = paste(img, mask, idx, offs)
    conf_px = F.interpolate(conf, size=img.shape[-2:], mode="nearest")
    return conf_px * copy + (1 - conf_px) * query


def topk_weights(Cn: torch.Tensor, k: int = 4, tau_k: float = 0.5):
    """ブロックごとに上位 k 候補とその重みを返す (単一候補コピーの神託上限を破るための拡張)。

    単一コピーは「argmin の1点」に賭けるので、その1点が正解からズレていれば
    誤差がそのまま残る。上位 k 候補を重み付き平均すれば、各候補が持つ独立な
    誤差成分が 1/k 近くまで減り、**神託 (正解を見て最良の1候補を選ぶ) の上限すら
    超えうる**。これは平均化そのものが持つ性質であり、選択の賢さとは別の力。

    `Cn`: (1,K,h,w) の全候補コスト。`idx_k`/`w_k`: (1,k,h,w)。

    重みは `block_confidence` と同じ「有効ブロックの cmin の中央値」でコストの
    絶対スケールを正規化してから `exp(-(c-cmin)/(tau_k*s))` を取る。画像ごと・
    テクスチャごとにコストの絶対スケールが変わるため、固定の tau では
    (a) 高周波テクスチャで重みが尖りすぎる、(b) 平坦な画像でほぼ一様になる、
    のどちらかに倒れる。中央値基準なら「その画像の中で相対的にどれだけ悪いか」で
    重みが決まる。

    無効候補 (コスト >= BIG*0.5, `solve_blocks` の番兵) は重み 0 にする。
    上位 k 個の中に無効候補が混ざりうる (穴の奥など有効候補が k 未満のブロック) ため。
    全 k 個が無効なブロックは、後段の `_fill_blocks_inward` 済みの単一 idx が
    使われるのと同じ扱いにするため、第0成分だけ重み 1・残り 0 にする。
    """
    BIG = 1e6
    c_k, idx_k = torch.topk(Cn, k, dim=1, largest=False)   # (1,k,h,w)
    valid_k = c_k < BIG * 0.5

    cmin = Cn.min(dim=1, keepdim=True).values
    matched = cmin < BIG * 0.5
    vals = cmin[matched]
    s = vals.median() if vals.numel() > 0 else torch.zeros((), dtype=Cn.dtype, device=Cn.device)
    if float(s) <= 0:
        s = torch.ones((), dtype=Cn.dtype, device=Cn.device)

    w = torch.exp(-(c_k - c_k[:, :1]) / (tau_k * float(s))) * valid_k.to(Cn.dtype)
    denom = w.sum(dim=1, keepdim=True) + 1e-12
    w_k = w / denom

    all_invalid = ~valid_k.any(dim=1, keepdim=True)        # (1,1,h,w)
    fallback_w = torch.zeros_like(w_k)
    fallback_w[:, 0:1] = 1.0
    w_k = torch.where(all_invalid.expand_as(w_k), fallback_w, w_k)
    return idx_k, w_k


def paste_topk(img: torch.Tensor, mask: torch.Tensor, idx_k: torch.Tensor,
              w_k: torch.Tensor, offs, block: int = BLOCK) -> torch.Tensor:
    """`paste` の複数候補版。上位 k 候補を重み付き平均で合成する。

    ランクごとに画素解像度へ最近傍拡大してからシフト画像を作り、重みを掛けて
    足し込む。`torch.unique` でそのランクに実際に現れるオフセットだけを回すので、
    2400 個の全オフセットを毎回舐めることはない (ランク数 k 回 × 実際に出現した
    オフセット数)。

    **契約は `paste` と同じ: マスク外にも合成結果が伸びる** (Stage D のポアソン膜が
    これを要求する)。`out = img.clone()` から始めて一部だけ上書きすると、初期値の
    `img` が残る画素が生じてマスク外の合成保証が崩れるので、重み付き和をそのまま
    全画素で作る (`w_k` の和は 1 に正規化済みなので、これは「近傍ブロックの
    オフセットで合成された値」を全画素に対して作ることになる)。
    """
    k = idx_k.shape[1]
    out = torch.zeros_like(img)
    for j in range(k):
        jf = F.interpolate(idx_k[:, j:j + 1].float(), size=img.shape[-2:],
                           mode="nearest").long()[0, 0]
        wf = F.interpolate(w_k[:, j:j + 1], size=img.shape[-2:], mode="nearest")
        for i in torch.unique(jf).tolist():
            oy, ox = offs[int(i)]
            m = (jf == i)[None, None]
            out = out + torch.where(m, _shift(img, oy, ox) * wf, torch.zeros_like(img))
    return out


def _blend(copy: torch.Tensor, query: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    """信頼度でコピー結果と query をブレンドする (`composite` から複製した内部核)。

    `composite` は内部で `paste` を直接呼ぶため、`paste_topk` の合成結果 `soft` を
    渡す経路が作れない。`composite` 自体は既存の挙動 (bit 一致) を変えたくないので、
    ブレンド部分だけをここに切り出し、`composite` と `inpaint` の両方から使う。
    """
    conf_px = F.interpolate(conf, size=copy.shape[-2:], mode="nearest")
    return conf_px * copy + (1 - conf_px) * query


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """分離可能な1次元ガウシアンで `x` をぼかす (`hf_transfer` の低周波抽出に使う)。

    `_box` (箱フィルタ) は Stage B の照合コストには十分だが、リング状の
    サイドローブを持ち「低周波/高周波」の境目がなだらかでない。`hf_transfer` は
    この境目そのものが移植するテクスチャの粒度を決めるので、なだらかに減衰する
    ガウシアンを使う。縦横に分離することで計算量を O(k) x2 に抑えるのは `_box` と
    同じ考え方。境界は `replicate` で pad する — ゼロ埋めだと縁で暗くなり、
    定数画像を通しても定数のままにならない (下の docstring にある「低周波は base
    に一致する」という性質が崩れる)。
    """
    radius = int(np.ceil(3.0 * sigma))
    xs = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
    out = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    out = F.conv2d(out, kx, groups=c)
    out = F.pad(out, (0, 0, radius, radius), mode="replicate")
    out = F.conv2d(out, ky, groups=c)
    return out


def hf_transfer(base: torch.Tensor, copy: torch.Tensor, mask: torch.Tensor,
                sigma: float = 2.0, alpha: float = 1.0) -> torch.Tensor:
    """`copy` の実画素から高周波成分だけを取り出し `base` に足す (SPEC 変更: 置き換え -> 移植)。

    **なぜ丸ごと置き換えないか**: `paste`/`paste_topk` は元画像の画素で丸ごと
    置き換えるため、`base` (StageF の出力) が稼いだ精度の高い低中周波まで
    コピー元の画素に差し替えてしまう。実測で「置き換え」は PSNR を StageF 単体
    より悪化させる一方 (k=1 で 23.49 dB、StageF 単体 24.66 dB)、grain/psd
    (テクスチャの質) は StageF 単体よりずっと良い (0.917 / 0.166)。つまり
    `base` は低中周波が正しく、`copy` は高周波が正しい。**周波数帯ごとに
    出処を分ければ両方の精度を同時に活かせる。**

    `copy` をガウシアンでぼかした `lo_copy` を「`copy` の低周波」とみなし、
    引くことで高周波 `hf = copy - lo_copy` を取り出す。`base` にそのまま足せば、
    `base` の低中周波は無傷のまま `copy` のテクスチャだけが上に乗る。`sigma` は
    低周波/高周波の境界 (大きいほど「高周波」とみなす帯域が狭くなる)、`alpha` は
    移植する強さ (1.0 で全量、0.0 で `base` のまま) を制御する。

    `mask` は受け取るが計算には使わない — `paste`/`paste_topk` と同じ契約
    (SPEC §4.4) で、**マスク外にも `base + alpha*hf` を伸ばして返す**。
    マスク外を `base` (または元画像) に戻すのは呼び出し側の責任であり、ここで
    `mask` を使って条件分岐すると、その責任の境界があいまいになる。引数として
    残すのは、呼び出し側 (`inpaint`) の他の合成関数 (`composite`/`_blend`) と
    シグネチャを揃え、将来ブロック単位でぼかし半径を変えるような拡張の余地を
    残すため。

    base  : (1,3,H,W) 低中周波の出処 (StageF の出力)
    copy  : (1,3,H,W) 高周波の出処 (`paste`/`paste_topk` の出力、実画素)
    mask  : (1,1,H,W) 1 = 穴 (契約の記録用。計算には使わない)
    戻り値 : (1,3,H,W)
    """
    del mask  # 計算には使わない (docstring 参照)。lint 用に明示的に握りつぶす。
    lo_copy = _gaussian_blur(copy, sigma)
    hf = copy - lo_copy
    return base + alpha * hf


def hole_distance(mask: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """穴の内部画素について、最も近い既知画素までの距離を近似する。

    **なぜ「穴の太さ」が要るか**: 用途val でマスク種別ごとに Stage D 後の PSNR を
    比べると、rect (太い矩形) だけコピーと StageF の優劣が逆転する:

        種別               コピー(K8)   StageF単体
        wire (細い線)        30.20        27.48
        dust (小さい点)      24.47        22.06
        stroke               22.13        21.34
        rect (太い矩形)      19.29        19.43   <- ここだけ逆転

    穴が太いと内部画素は既知画素から遠く、コピー元として正しい実画素が存在しない
    (目視でも rect 内部に 32px の正方形パッチワークやにじみを確認済み)。この
    「既知画素からの遠さ」を画素ごとに測るために距離が要る。

    **なぜ厳密な距離変換ではなく膨張の反復で近似するか**: scipy の distance
    transform を使えば厳密だが依存が増える。ここでは `F.max_pool2d` による
    3x3 膨張を繰り返し、既知領域からの Chebyshev 距離 (帯状の穴では実際の距離と
    一致する) を「何回目の反復で到達したか」として近似する。ベクトル化されており
    画素単位の Python ループは無い (反復回数ぶんのループのみ)。

    mask    : (1,1,H,W) 1 = 穴
    max_iter: 膨張の最大反復回数。到達しない画素 (穴が max_iter より厚い) には
              この値をそのまま入れる。
    戻り値  : (1,1,H,W) float。マスク外 (mask==0) は契約により常に 0。
    """
    is_hole = (mask > 0.5).to(mask.dtype)
    reached = 1.0 - is_hole                     # 既知画素は反復 0 で到達済み
    dist = torch.zeros_like(mask)
    frontier = reached
    for i in range(1, max_iter + 1):
        dilated = F.max_pool2d(F.pad(frontier, (1, 1, 1, 1)), 3, stride=1)
        newly = (dilated - reached).clamp(0, 1)
        if float(newly.sum()) == 0:
            break
        dist = dist + newly * i
        reached = (reached + newly).clamp(0, 1)
        frontier = dilated
    unreached = is_hole * (1.0 - reached)
    dist = dist + unreached * max_iter
    return dist * is_hole


def adaptive_blend(copy: torch.Tensor, base: torch.Tensor, mask: torch.Tensor,
                   d0: float = 8.0, d1: float = 24.0) -> torch.Tensor:
    """穴の太さに応じて `copy` (ブロックコピー) と `base` (StageF) を画素単位で混ぜる。

    `hole_distance` の docstring にある通り、用途val で rect (太い矩形) だけ
    StageF 単体 (19.43 dB) がコピー (19.29 dB) に僅かに勝つ。穴が太いほど内部
    画素は既知画素から遠く、コピー元として正しい実画素が存在しない。逆に細い穴
    (wire 30.20 vs 27.48 など) は周囲の実画素で足りるためコピーが強い。

    **処方**: `hole_distance` で測った「最も近い既知画素までの距離 d」を使い、
    d0 以内 (既知画素から近い = 細い穴の縁や小さい穴) は copy をそのまま、
    d1 を超える (太い穴の内部) は base に完全に委ね、その間は線形に混ぜる。
    `w = clamp((d - d0) / (d1 - d0), 0, 1)` は 0 で完全に copy、1 で完全に base。

    **マスク外の扱い**: `hole_distance` はマスク外で d=0 を返す契約なので、
    マスク外では w=0 になり copy がそのまま出る。これは `paste` と同じ
    「マスク外にも合成が伸びる」契約 (§4.4) を保つために正しい —
    マスク外を base や img に戻すのはここでは行わない (呼び出し側の責任)。
    """
    d = hole_distance(mask)
    w = torch.clamp((d - d0) / (d1 - d0), 0.0, 1.0)
    return (1.0 - w) * copy + w * base


def _gauss_pyr(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """ガウシアンピラミッドを作る (`_lap_pyr`/`_collapse` の土台)。

    ダウンサンプルは `F.avg_pool2d(x, 2)` のみを使う。**`mode="area"` の
    `F.interpolate` は絶対に使わない** — MPS の adaptive average pool は
    入力サイズが出力サイズで割り切れる場合しか実装されておらず、これで実際に
    学習ジョブが落ちたことがある (`lowfreq_query` は既存の `mode="area"` 呼び出し
    のまま残しているが、新規コードでは踏まない)。`avg_pool2d(x, 2)` は kernel=2/
    stride=2 の固定形状で MPS でも問題なく動く。

    `pyr[0]` は `x` そのもの、`pyr[i]` は `pyr[i-1]` を半分に縮めたもの。
    奇数サイズは切り捨てられる (131 -> 65 など) が、`_lap_pyr`/`_collapse` は
    常に「一つ細かい層の実際のサイズ」を明示して upsample するため、
    情報を失わずに往復できる (下記 `_lap_pyr` 参照)。
    """
    pyr = [x]
    cur = x
    for _ in range(1, levels):
        cur = F.avg_pool2d(cur, 2)
        pyr.append(cur)
    return pyr


def _lap_pyr(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """ラプラシアンピラミッド。各レベルが「その解像度だけが持つ周波数帯」を表す。

    `lap[i] = gauss[i] - upsample(gauss[i+1], size=gauss[i].shape)` (i < levels-1)、
    最後のレベル `lap[levels-1]` はガウシアンピラミッドの最粗解像度そのもの
    (低周波の残差)。`band_composite` はここで作った帯ごとに `base`/`copy` の
    どちらを使うかを切り替える。

    upsample は必ず `F.interpolate(..., size=<一つ細かい層の実際の空間サイズ>,
    mode="bilinear", align_corners=False)` を使う。`scale_factor` 指定や
    `mode="area"` は使わない — 前者は奇数サイズで丸め誤差が蓄積して往復が
    ズレる原因になり、後者は `_gauss_pyr` と同じ理由で MPS で壊れる。
    """
    g = _gauss_pyr(x, levels)
    lap = []
    for i in range(levels - 1):
        up = F.interpolate(g[i + 1], size=g[i].shape[-2:],
                           mode="bilinear", align_corners=False)
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def _collapse(lap: list[torch.Tensor]) -> torch.Tensor:
    """`_lap_pyr` の逆演算。最粗レベルから積み上げて元の解像度へ戻す。

    `_lap_pyr(x, n)` の各レベルは `gauss[i] - upsample(gauss[i+1])` として
    作られているので、粗い方から `upsample() + lap[i]` を繰り返すだけで
    厳密に `gauss[i]` (最終的には `x`) が復元される。ここでも upsample の
    ターゲットサイズは常に対応する `lap[i]` の実サイズを明示する
    (`_lap_pyr` と同じ理由。奇数サイズでも往復が厳密に一致する)。
    """
    cur = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        cur = F.interpolate(cur, size=lap[i].shape[-2:],
                            mode="bilinear", align_corners=False) + lap[i]
    return cur


def photometric_align(copy: torch.Tensor, base: torch.Tensor, mask: torch.Tensor,
                      block: int = BLOCK) -> torch.Tensor:
    """貼ったブロックが周囲より明るく浮く問題への対処 (加算のみのブロック単位露出補正)。

    目視検証で確認された破綻その2: 貼ったブロックが周囲より明るく浮き、四角い
    明度の段差として見える。`copy` (ブロックコピー) はコピー元ブロックの平均輝度を
    そのまま持ち込むため、コピー先の周囲の明るさとズレることがある。

    **加算補正のみで、乗算やヒストグラム一致はしない。** 乗算やヒストグラム一致は
    ブロック内のコントラスト比そのものを変えてしまい、テクスチャの粒状感を壊す
    (過剰補正)。単純な加算オフセットなら、ブロックの平均だけをずらしてテクスチャの
    相対的な起伏はそのまま保たれる。

    ブロック解像度 (`avg_pool2d(base, block) - avg_pool2d(copy, block)`) で
    補正量を求め、**画素解像度へ bilinear で滑らかに拡大してから** `copy` に足す。
    最近傍で拡大すると補正量自体がブロック格子状に四角く見えてしまい、結局
    「段差の場所が変わっただけ」になる (`paste_feather` の失敗と同じ理屈:
    不連続を残したまま処理すると不連続のまま残る)。bilinear ならブロック境界を
    またいで補正量が滑らかに変化するので、段差そのものが解消される。

    `mask` は受け取るが計算には使わない (`hf_transfer` と同じ契約)。**マスク外にも
    合成が伸びる**契約 (SPEC §4.4) を保つため、全画素に対して補正量を計算する —
    マスク外を元に戻すのは呼び出し側の責任。
    """
    del mask  # 計算には使わない (docstring 参照)。lint 用に明示的に握りつぶす。
    diff_block = F.avg_pool2d(base, block) - F.avg_pool2d(copy, block)
    diff_px = F.interpolate(diff_block, size=copy.shape[-2:],
                            mode="bilinear", align_corners=False)
    return copy + diff_px


def band_composite(base: torch.Tensor, copy: torch.Tensor, mask: torch.Tensor,
                   levels: int = 5, cross: int = 2,
                   d0: float = 4.0, d1: float = 16.0) -> torch.Tensor:
    """周波数帯ごとに供給源を切り替える多重解像度合成 (ラプラシアンピラミッド)。

    目視検証で確認された破綻その1: 隣接ブロックが別々の場所からコピーされ、境界で
    模様が食い違う 32px 格子が肉眼で見える。

    **過去に試して失敗した対処 (同じ轍を踏まない):**
      - `paste_feather` (一律の羽化): 段差エネルギーは 75% 減ったが、格子が
        **にじみ**に化けただけだった。1ケースは 42.3 -> 18.0 dB と壊滅した。
        **全周波数を同じ幅で混ぜたのが原因** — 境界だけでなくブロック内部の
        テクスチャまで一緒にぼかしてしまう。
      - `hf_transfer` (単一スケールの高周波移植): psd 0.1037 -> 0.1228 と悪化、
        boundary も 1.08 -> 1.77 と悪化した。単一のガウシアン sigma で
        「高周波/低周波」を一度に切るのは境界という現象そのものに対しては
        荒すぎる。

    **なぜ帯ごとに分けると解決するか**: `base` (StageF の全解像度出力) は
    配置が正しいがぼけている、`copy` (ブロックコピー) はテクスチャが正しいが
    配置がブロックごとにズレる。実測 grain: base 0.109 / copy 1.186
    (理想 1.0)。base は PSNR で LaMa と互角。**ブロックの継ぎ目・浮きは低〜中
    周波の「配置のズレ」であり、細かいテクスチャの粒状感は高周波にしか無い。**
    したがって粗い帯 (`_lap_pyr` のレベル `l >= cross`) は `base`、
    細かい帯 (`l < cross`) は `copy` から取れば、`base` の正しい配置と `copy` の
    正しいテクスチャを両方活かしたまま、境界の食い違い (低〜中周波成分) だけを
    `base` の滑らかな配置に置き換えて消せる。`paste_feather` のように空間的に
    ぼかすのではなく、**周波数で** 出処を切り替えるので、ブロック内部の高周波
    テクスチャは (空間方向の重みが 0 の限り) 一切ぼかさずに済む。

    境界のレベル (`l == cross`) は `base`/`copy` を 0.5 ずつ混ぜる。ここを
    ハード切り替えにすると、隣接レベル間で急に出処が変わり、その帯自体の
    振幅差が新たな (低次の) 継ぎ目を生みかねないため。

    **さらに空間方向の重みを掛ける** (`hole_distance` を再利用): 細かい帯
    (`l < cross`) についてのみ `w = clamp((d - d0)/(d1 - d0), 0, 1)` の
    `(1 - w)` を掛ける。穴の奥 (`d` が大きい) では `copy` の高周波を信用しない
    — 周囲の実画素から遠いブロックほどコピー元の一致に根拠が無いため
    (`adaptive_blend` と同じ理由づけ)。重みはレベルごとの解像度へ bilinear で
    縮小してから使う (`photometric_align` と同じ理由で、最近傍だと重みの
    境目自体が四角く見える)。

    最後に `_collapse` して元解像度へ戻す。**マスク外にも合成が伸びる契約**
    (SPEC §4.4) を保つ — `base`/`copy` はどちらも全画素に対して計算されている
    ものをそのまま渡す前提で、ここでもマスクによる画素単位の条件分岐はしない。

    base  : (1,3,H,W) 低中周波の出処 (StageF の出力など、配置が正しい)
    copy  : (1,3,H,W) 高周波の出処 (`paste`/`paste_topk` などの実画素コピー)
    mask  : (1,1,H,W) 1 = 穴。`hole_distance` にそのまま渡す。
    戻り値 : (1,3,H,W)
    """
    lap_base = _lap_pyr(base, levels)
    lap_copy = _lap_pyr(copy, levels)
    d = hole_distance(mask)
    w = torch.clamp((d - d0) / (d1 - d0), 0.0, 1.0)

    out_lap = []
    for l in range(levels):
        size = lap_base[l].shape[-2:]
        if l < cross:
            wl = F.interpolate(w, size=size, mode="bilinear", align_corners=False)
            out_lap.append(lap_copy[l] * (1.0 - wl))
        elif l == cross:
            out_lap.append(0.5 * lap_base[l] + 0.5 * lap_copy[l])
        else:
            out_lap.append(lap_base[l])
    return _collapse(out_lap)


def inpaint(img: torch.Tensor, mask: torch.Tensor, feat=None,
           fallback: bool = False, tau_c: float = 1.0,
           topk: int = 1, tau_k: float = 0.5,
           coherent: bool = False, lam: float = 0.5, tol: float = 8.0,
           coh_iters: int = 4, passes: int = 1,
           hf_only: bool = False, hf_sigma: float = 2.0, hf_alpha: float = 1.0,
           feather: float = 0.0,
           adaptive: bool = False, d0: float = 8.0, d1: float = 24.0,
           band: bool = False, levels: int = 5, cross: int = 2,
           photo: bool = False,
           **kw) -> torch.Tensor:
    """Stage B の入口。

    `fallback=True` かつ `query` (Stage A の低周波補完) が与えられた場合のみ、
    信頼度ゲート合成 (`block_confidence` + `composite`) の経路を通る。それ以外は
    従来どおり `paste` をそのまま返す (bit 一致)。**デフォルトを従来どおりにするのは、
    fallback 経路が query を要求するため、query の無い既存呼び出しを壊さないため。**

    `topk` >= 2 のとき、単一候補の `paste` の代わりに上位 `topk` 候補の重み付き
    平均 `paste_topk` を使う (`topk_weights` 参照)。`topk <= 1` では従来と完全に
    同一の経路 (`paste` / `composite`) を通り、既存呼び出しは bit 一致のまま変わらない。

    `coherent=True` のとき、`solve_blocks` 直後に `coherent_idx` を挟んでブロック
    ごとの選択を近傍と整合させる (`coherent_idx` 参照)。それ以降の経路
    (paste / composite / topk / fallback) はそのまま変わらない。**既定は
    `coherent=False` で、この場合は従来と完全に同一の経路 (bit 一致) を通る。**

    `passes` >= 2 のとき、ブートストラップ再探索 (multi-pass) を行う。`query`
    (Stage A の低周波補完) は**ぼけている**ため、1パス目の照合コストは低周波
    成分しか見ておらず、高周波テクスチャで候補を判別できない。1パス目の出力
    `paste(...)` は実画素のコピーなので高周波テクスチャを持つ。これを2パス目の
    クエリに差し替えれば、以降のパスは高周波を含むコストで候補を選び直せる
    (古典的なマルチパス・テクスチャ合成)。中間パスは常に素の `paste` を使い、
    最終パスだけ `coherent` / `topk` / `fallback` の既存分岐を適用する。**平均化は
    しない** — 中間パスも `paste` (1ブロック1候補のコピー) を通すので、テクスチャを
    ぼかす操作は挟まらない。

    `passes` は `query` が与えられている場合のみ有効。初期クエリが無いと穴の
    内部を照合できず (`solve_blocks` の docstring 参照)、ブートストラップの
    出発点が無いため、`query` が無いときは `passes` を無視して従来の経路を通る
    (既存呼び出しは bit 一致のまま変わらない)。

    `hf_only=True` かつ `query` が与えられている場合、`paste`/`paste_topk` の
    コピー結果を丸ごと返す代わりに `hf_transfer` を通す。**置き換えではなく
    高周波だけの移植にする** (SPEC の設計変更、`hf_transfer` の docstring 参照)。
    `query` (StageF の出力など) の低中周波はそのまま保ち、コピー結果からは
    高周波成分だけを足すことで、StageF の精度とコピーのテクスチャを両取りする。
    `hf_only=True` のときは `fallback` (信頼度ゲート) の分岐には入らない —
    両方を同時に有効にした場合の合成順序は未定義のため、呼び出し側で選ばせる
    のではなく `hf_only` を優先して確定させる。`query` が無い場合は `base` が
    作れず `hf_transfer` が成立しないため `hf_only` を無視し、従来の経路を通る
    (既存呼び出しは bit 一致のまま変わらない)。`hf_sigma`/`hf_alpha` は
    `hf_transfer` にそのまま渡す。

    `feather > 0` かつ `topk <= 1` のとき、単一候補の貼り付けに `paste` の代わりに
    `paste_feather(..., overlap=feather)` を使い、`block` 格子の継ぎ目を重なり +
    ハン窓で滑らかに混ぜる (`paste_feather` の docstring 参照)。`hf_only`/
    `fallback` のどちらとも組み合わせられる — その場合もコピー結果を作る箇所だけ
    `paste_feather` に差し替え、以降の合成 (`hf_transfer`/`composite` 相当の
    ブレンド) はそのまま適用する。`feather <= 0` のときは `paste` を呼ぶ従来の
    経路をそのまま通るため bit 一致を保つ。`topk >= 2` は既にブロックを重み付き
    平均しており、`paste_feather` の重ね貼りと二重に平均する意味が無いため、
    `feather > 0` かつ `topk >= 2` のときは `feather` を無視して従来の topk 経路
    (`paste_topk`) を通る。

    `adaptive=True` かつ `query` が与えられている場合、コピー結果 (`paste`/
    `paste_topk`/`paste_feather` — 分岐は上記と同じ) をそのまま返す代わりに
    `adaptive_blend(copy, query, mask, d0, d1)` を返す。**用途val でマスク種別
    ごとの PSNR (Stage D 後) を見ると rect (太い矩形, 19.29 dB) だけコピーが
    StageF 単体 (19.43 dB) に負ける** (wire 30.20/27.48、dust 24.47/22.06、
    stroke 22.13/21.34 はコピーが勝つ) — 穴が太いと内部画素は既知画素から遠く、
    コピー元として正しい実画素が存在しないため (`hole_distance` の docstring
    参照)。`adaptive_blend` は穴の太さ (最も近い既知画素までの距離) で画素ごとに
    copy と query (StageF などの推測) を混ぜ、この逆転を画素単位で解消する。
    `adaptive=True` のときは `hf_only`/`fallback` の分岐には入らない — `hf_only`
    と同様、複数の合成を同時に有効にした場合の順序は未定義のため、`adaptive`
    を優先して確定させる。`query` が無い場合は `adaptive` を無視して従来の経路
    を通る (既存呼び出しは bit 一致のまま変わらない)。`adaptive=False` (既定) の
    ときは以下の分岐に一切触れないため、既存呼び出しは完全に bit 一致を保つ。

    `band=True` かつ `query` が与えられている場合、コピー結果を `adaptive_blend`
    の代わりに `band_composite(base=query, copy=<コピー結果>, mask, levels, cross,
    d0, d1)` で合成する。目視検証で確認された「32px 格子」と「貼ったブロックの
    浮き」の両方に対する本命の対処 (`band_composite` の docstring 参照。
    `paste_feather`/`hf_transfer` が単一の混ぜ幅・単一スケールで失敗した経緯を
    踏まえ、周波数帯ごとに `base`/`copy` の出処を切り替える)。`band=True` の
    ときは `adaptive`/`hf_only` の分岐には入らない — 複数の合成を同時に有効に
    した場合の順序は未定義のため、他のフラグと同様に `band` を優先して確定
    させる (`fallback` の分岐にも入らない。`band` はそれらと同じ「コピー結果を
    最終的に何と混ぜるか」を選ぶスイッチであり、`band` が最優先というのが
    ここでの唯一の解釈)。`levels`/`cross`/`d0`/`d1` はそのまま `band_composite`
    に渡す。`query` が無い場合は `base` が作れないため `band` を無視して従来の
    経路を通る (既存呼び出しは bit 一致のまま変わらない)。`band=False` (既定)
    のときは以下の分岐に一切触れないため、既存呼び出しは完全に bit 一致を保つ。

    `photo=True` のとき、コピー結果 (`paste`/`paste_topk`/`paste_feather`) に
    `photometric_align(copy, query, mask)` を適用してから、上記の各分岐 (band/
    adaptive/hf_only/fallback/そのまま返す) に渡す。「貼ったブロックが周囲より
    明るく浮く」問題への対処 (`photometric_align` の docstring 参照)。`query`
    (StageF の出力など、正しい明るさの基準) が無い場合は補正の基準が無いため
    `photo` を無視する。`photo` と `band` は併用できる — その場合は
    **photo で明るさを揃えてから band で周波数帯ごとに合成する** (photo -> band
    の順)。`photo=False` (既定) のときは以下の分岐に一切触れないため、
    既存呼び出しは完全に bit 一致を保つ。
    """
    query = kw.get("query")
    if passes is not None and passes >= 2 and query is not None:
        q = query
        for _ in range(passes - 1):
            idx_mid, offs_mid, _ = solve_blocks(img, mask, feat, **{**kw, "query": q})
            out_mid = paste(img, mask, idx_mid, offs_mid)
            # paste はマスク外にも合成が伸びる契約 (§4.4)。次パスのクエリでは
            # マスク外を必ず元画像に戻す。ここを戻さないと誤差が累積する。
            q = torch.where(mask > 0.5, out_mid, img)
        kw = {**kw, "query": q}

    idx, offs, Cn = solve_blocks(img, mask, feat, **kw)
    if coherent:
        idx = coherent_idx(Cn, offs, idx, lam, tol, coh_iters)
    query = kw.get("query")   # multi-pass 経路では最終パスに使ったクエリに更新済み

    if topk is None or topk <= 1:
        # feather <= 0 のときは以下の分岐がすべて元のまま `paste` を呼ぶだけなので
        # 既存呼び出しは bit 一致を保つ。feather > 0 のときだけ `paste_feather` に
        # 差し替える (topk >= 2 の経路には触れないため、topk 併用時は無視される)。
        use_feather = feather > 0

        def _copy_out() -> torch.Tensor:
            return (paste_feather(img, mask, idx, offs, overlap=feather)
                   if use_feather else paste(img, mask, idx, offs))

        # band は adaptive/hf_only/fallback より先に見る (docstring 参照)。
        # `query is not None` が偽なら何もせず素通りし、以降の従来分岐へ落ちる
        # ので band=False (既定) 呼び出しの bit 一致は崩れない。
        if band and query is not None:
            copy_out = _copy_out()
            if photo:
                copy_out = photometric_align(copy_out, query, mask)
            return band_composite(query, copy_out, mask, levels, cross, d0, d1)
        if adaptive and query is not None:
            copy_out = _copy_out()
            if photo:
                copy_out = photometric_align(copy_out, query, mask)
            return adaptive_blend(copy_out, query, mask, d0, d1)
        if hf_only and query is not None:
            copy_out = _copy_out()
            if photo:
                copy_out = photometric_align(copy_out, query, mask)
            return hf_transfer(query, copy_out, mask, hf_sigma, hf_alpha)
        if fallback and query is not None:
            conf = block_confidence(Cn, tau_c)
            if photo:
                copy_out = _copy_out()
                copy_out = photometric_align(copy_out, query, mask)
                return _blend(copy_out, query, conf)
            if use_feather:
                copy_out = paste_feather(img, mask, idx, offs, overlap=feather)
                return _blend(copy_out, query, conf)
            return composite(img, mask, query, idx, offs, conf)
        if photo and query is not None:
            return photometric_align(_copy_out(), query, mask)
        if use_feather:
            return paste_feather(img, mask, idx, offs, overlap=feather)
        return paste(img, mask, idx, offs)

    idx_k, w_k = topk_weights(Cn, topk, tau_k)
    # **全候補が無効なブロック (穴の奥) は `solve_blocks` が近傍から伝播させた
    # `idx` を使う。** `topk_weights` はそういうブロックで第0候補に重み1を置くが、
    # その第0候補は「BIG の中で最小」= 穴の中からコピーする無効オフセットである。
    # 単一候補の経路は `_fill_blocks_inward` 済みの `idx` を使っており、
    # ここを揃えないと topk のときだけ穴の内部を複製してしまう。
    dead = ~(Cn.min(dim=1, keepdim=True).values < 5e5)
    idx_k = torch.where(dead.expand_as(idx_k), idx.expand_as(idx_k), idx_k)
    soft = paste_topk(img, mask, idx_k, w_k, offs)
    # `photo`/`band` は既定 False なので、以下の `if photo:` は既存呼び出しでは
    # 決して真にならず、`soft` はここまでの計算のまま従来の分岐へ渡る
    # (bit 一致を保つ)。
    if photo and query is not None:
        soft = photometric_align(soft, query, mask)
    if band and query is not None:
        return band_composite(query, soft, mask, levels, cross, d0, d1)
    if adaptive and query is not None:
        return adaptive_blend(soft, query, mask, d0, d1)
    if hf_only and query is not None:
        return hf_transfer(query, soft, mask, hf_sigma, hf_alpha)
    if fallback and query is not None:
        conf = block_confidence(Cn, tau_c)
        return _blend(soft, query, conf)
    return soft


def lowfreq_query(img: torch.Tensor, mask: torch.Tensor, fill: torch.Tensor,
                  shrink: int = 8) -> torch.Tensor:
    """補完結果 `fill` の低周波だけを取り出して穴を埋める。

    高周波まで使うと、照合が「補完結果のテクスチャ」に引きずられる。
    構造 (意味) だけを借り、実画素は元画像から取るのが §3.1 の分業。
    """
    lo = F.interpolate(F.interpolate(fill, scale_factor=1.0 / shrink, mode="area"),
                       size=fill.shape[-2:], mode="bilinear", align_corners=False)
    return torch.where(mask > 0.5, lo, img)


def inpaint_np(p: np.ndarray, mask: np.ndarray, device: str = "mps",
               feat=None, **kw) -> np.ndarray:
    """`inpaint` の numpy ラッパー。

    `fallback` / `tau_c` は `**kw` 経由でそのまま `inpaint` へ透過する
    (シグネチャに明示引数を増やす必要が無い)。`query` を渡す場合は
    呼び出し側で既に `device` 上のテンソルへ変換済みのものを渡すこと
    (`p`/`mask` と違って numpy -> tensor 変換をここでは行わない)。
    """
    x = torch.from_numpy(p.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    m = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
    with torch.inference_mode():
        y = inpaint(x, m, feat, **kw)
    return y[0].permute(1, 2, 0).float().cpu().numpy()
