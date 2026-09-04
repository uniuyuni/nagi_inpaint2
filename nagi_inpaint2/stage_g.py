"""Stage G — 残差 (StageF の取りこぼし) を Rectified Flow で標本として引く (G1 検証)。

**解こうとしている問題**: `nagi_inpaint2/stage_f.py` の `StageF` は L1 損失で学習しており、
穴の内部で**条件付き平均**を出す。結果、人物サイズの穴に**ぼけた塊**が残る。
盲検A/Bでユーザに判定してもらった結果は **0勝16敗4分 (p=0.00003)** で、
敗因は「ほとんどのケースでぼけた塊」「ところどころ四角い継ぎ目」だった。

**効く理屈**: 残差 `r = 正解 - StageF出力` を**生成モデルで標本として引く**。
平均を出さないので、ぼけない。Rectified Flow (Flow Matching) を使うと
目的関数が単なる回帰 (速度場の L2) なので、**学習は L1 と同程度に安定**する
(GAN は2回試して崩壊した。WTA は多様性は出たが各仮説が依然 L1 でぼけた)。

**実測で確認済みの前提** (eval/residual_stats.py, 人物val 24ケース):
  - 残差は白色雑音ではない: 自己相関 lag1 = **+0.911**、PSD の低域/高域比 **175倍**
  - 残差の振幅は元画像の高周波の **3.24倍** → **高周波だけでなく中間周波数も欠けている**。
    したがって残差の帯域を高周波に限定してはいけない (このモジュールは r をそのまま
    3ch のフルカラー残差として扱い、ハイパス等の帯域制限を一切かけない)。
  - Stage B の検索パッチとの相関は **+0.210** (無作為対照 +0.009) →
    条件として与える価値はあるが**決定的ではない**。モデルが無視できる補助条件にすること
    (`copy_resid` チャンネルは他条件と対等な入力チャンネルにするだけで、特別扱いしない)。
  - 穴の奥ほど残差が大きい (縁 0.044 → 32px 以上 0.126) → `dist` (穴の内部距離) を
    条件に含める根拠。

**効かない条件 (検出方法)**: サンプリングした残差が穴の内部で**高周波を持たない**なら、
この設計は成立しない。検出方法は「`StageF出力 + サンプリング残差` の grain 比が
`StageF` 単体より明確に上がるか」。grain は `train/train_stage_fw.py` の
`grain_ratio_torch` を import して使う (これは `eval/metrics.py` の numpy 版と数値一致
することを検証済み)。**細いマスクでは grain が nan になり盲目になる**ので、
`train/train_stage_g.py` の検証は必ず人物マスクで測る (このファイル自体は形状の
検証のみ行う)。

**なぜ注意機構を使わない畳み込み UNet か**: この段階の目的は「実現可能性の一次
スクリーニング」であり、モデルを大きくすることが目的ではない。文献 (Cherel et al.)
では 160K パラメータでも Rectified Flow によるテクスチャ復元が成立している。
小ささそのものが「この設計が効くなら小さくても効くはず」という検証の一部なので、
パラメータ数は 3M 以下に抑える (実装は ~1.7M 程度に収まる)。

**正規化は InstanceNorm。BatchNorm は使わない。** マスク入力で統計が壊れる失敗が
StageA 開発時に既に一度あった (StageF の docstring 参照)。Stage G でも同じ罠を
踏まないよう、`stage_f.py` / `stage_a.py` と同じ選択をする。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage_b_search import _gaussian_blur
from .stage_b_simple import _box

TIME_DIM = 128    # 正弦波時刻埋め込みの次元


class SinusoidalTimeEmbedding(nn.Module):
    """時刻 t (0..1 のスカラー) を正弦波埋め込みにする (Transformer 位置埋め込みと同じ式)。"""

    def __init__(self, dim: int = TIME_DIM):
        super().__init__()
        assert dim % 2 == 0, "sin/cos で半分ずつ使うため偶数にする"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (N,) -> (N, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1))
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _ConvINReLU(nn.Module):
    """畳み込み + InstanceNorm + 時刻埋め込みの加算 (FiLM の簡素版) + ReLU。

    「時刻 t は正弦波埋め込み -> MLP -> 各段に加算 (FiLM でもよいが簡素に加算で可)」
    という仕様どおり、スケールではなくバイアスとして channel ごとに加算するだけにする。
    """

    def __init__(self, cin: int, cout: int, stride: int = 1, time_dim: int = TIME_DIM):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm2d(cout, affine=True)
        self.time_proj = nn.Linear(time_dim, cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = x + self.time_proj(temb)[:, :, None, None]
        return self.act(x)


class StageG(nn.Module):
    """残差 `r` を予測する小さな条件付き速度場モデル (Rectified Flow の `v_theta`)。

    forward(z, t, cond) -> v
      z    : (N,3,H,W) 現在の状態 (ノイズと残差の間を補間した点)
      t    : (N,)       時刻 0..1
      cond : (N,C,H,W)  条件 (`make_cond` 参照。既定 cond_ch=8)
      戻り値: (N,3,H,W) 予測速度

    構成: stem (全解像度) -> エンコーダ3段 (stride 2 x3) -> 中間ブロック (1/8解像度) ->
          デコーダ3段 (全解像度に復元)。**注意機構は使わない** (単純な畳み込み UNet)。
          `stage_f.py` の StageF と同じ skip 接続・nearest 補間の書き方を踏襲する
          (`F.interpolate(..., mode="area")` は MPS で落ちるため使わない)。
    """

    ENC_DEPTH = 3

    def __init__(self, base: int = 32, cond_ch: int = 8):
        super().__init__()
        self.base = base
        self.cond_ch = cond_ch

        self.time_embed = SinusoidalTimeEmbedding(TIME_DIM)
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_DIM, TIME_DIM * 2), nn.SiLU(),
            nn.Linear(TIME_DIM * 2, TIME_DIM))

        in_ch = 3 + cond_ch
        self.stem = _ConvINReLU(in_ch, base)
        enc, ch = [], base
        for _ in range(self.ENC_DEPTH):
            enc.append(_ConvINReLU(ch, ch * 2, stride=2))
            ch *= 2
        self.enc = nn.ModuleList(enc)
        self.mid = _ConvINReLU(ch, ch)
        dec = []
        for _ in range(self.ENC_DEPTH):
            dec.append(_ConvINReLU(ch + ch // 2, ch // 2))
            ch //= 2
        self.dec = nn.ModuleList(dec)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(self.time_embed(t))
        x = torch.cat([z, cond], dim=1)
        x = self.stem(x, temb)
        skips = [x]
        for e in self.enc:
            x = e(x, temb)
            skips.append(x)
        x = self.mid(x, temb)
        for d in self.dec:
            skip = skips[-2]
            skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = d(torch.cat([x, skip], dim=1), temb)
        return self.out(x)


def make_cond(sf_out: torch.Tensor, mask: torch.Tensor, dist: torch.Tensor,
              copy_resid: torch.Tensor) -> torch.Tensor:
    """条件テンソル (N,8,H,W) を組み立てる。

    **チャンネル順は固定**:
      ch 0-2 : `sf_out`      StageF出力 (知覚空間3ch)
      ch 3   : `mask`        1 = 穴
      ch 4   : `dist / 32`   穴の内部距離 (`stage_b_search.hole_distance`) を 32 で
                              割って [0,1] にクリップしたもの。32px は Stage B の
                              既定ブロックサイズに合わせた正規化スケール
      ch 5-7 : `copy_resid`  Stage B の検索パッチ残差 (copy - sf_out)。検索を
                              行わない場合は 0 で埋める (`train/train_stage_g.py`
                              の `--use-copy-cond 0`)

    `rectified_flow_loss` はこの並びを前提に ch 3 をマスクとして読む。
    """
    dist_n = (dist / 32.0).clamp(0.0, 1.0)
    return torch.cat([sf_out, mask, dist_n, copy_resid], dim=1)


def rectified_flow_loss(model: StageG, r: torch.Tensor, cond: torch.Tensor,
                        rng: torch.Generator | None = None):
    """Rectified Flow (Flow Matching) の学習損失。

    `t ~ U(0,1)`、`noise ~ N(0,I)`、`z_t = (1-t)*noise + t*r`、目標速度
    `v = r - noise`。損失は**マスク領域に重み付けした L2**:
    `((model(z_t,t,cond) - v)**2 * mask).sum() / (mask.sum()+eps)`
    (穴の外は学習しない。Cherel et al. の weighted L2 と同じ流儀)。

    マスクは `cond` の ch3 (`make_cond` のチャンネル順) からそのまま読む —
    別引数にせず `cond` に一本化することで、呼び出し側が違うマスクを渡して
    条件と損失がずれる事故を防ぐ。

    `rng`: 再現性が要る場合に渡す `torch.Generator` (CPU 上で作ること。MPS の
    Generator は torch が正式サポートしていないため、CPU で乱数を作ってから
    `cond`/`r` のデバイスへ転送する)。None なら通常の乱数を使う。

    戻り値: (loss, parts)。`parts` は監視用の dict。
    """
    n = r.shape[0]
    device = r.device
    if rng is not None:
        t = torch.rand(n, generator=rng).to(device)
        noise = torch.randn(r.shape, generator=rng).to(device)
    else:
        t = torch.rand(n, device=device)
        noise = torch.randn_like(r)

    t_b = t.view(n, 1, 1, 1)
    z_t = (1.0 - t_b) * noise + t_b * r
    v_target = r - noise
    v_pred = model(z_t, t, cond)

    mask = cond[:, 3:4]                     # make_cond のチャンネル順に依存 (固定)
    m = mask.expand_as(v_pred)
    diff2 = (v_pred - v_target) ** 2 * m
    loss = diff2.sum() / (m.sum() + 1e-6)

    parts = dict(t_mean=float(t.mean()), hole_frac=float(m.mean()),
                v_pred_std=float(v_pred.detach().std()),
                v_target_std=float(v_target.std()))
    return loss, parts


def sample_residual(model: StageG, cond: torch.Tensor, mask: torch.Tensor,
                    steps: int = 8, generator: torch.Generator | None = None) -> torch.Tensor:
    """ノイズから残差を復元する。オイラー法で `steps` 回積分する。

    `z=noise (t=0)` から出発し、`z = z + v*dt` を `t=0 -> 1` まで進める
    (Rectified Flow の推論そのもの)。`generator` を渡せば決定的に再現できる
    (`rectified_flow_loss` と同じ理由で CPU の Generator を想定する)。

    **マスク外は 0 にして返す** — 残差は穴の中だけの量であり、`stage_f.py` の
    forward と違ってこの関数は呼び出し側にマスク合成を委ねない (仕様どおり)。
    """
    n, _, h, w = cond.shape
    device = cond.device
    if generator is not None:
        z = torch.randn(n, 3, h, w, generator=generator).to(device)
    else:
        z = torch.randn(n, 3, h, w, device=device)

    dt = 1.0 / steps
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((n,), i * dt, device=device)
            v = model(z, t, cond)
            z = z + v * dt
    if was_training:
        model.train()

    return z * (mask > 0.5).float()


# ============================================================================
# 残差の局所スケール正規化 (G2)
#
# **効く理屈と、効かない条件 (引用元: このタスクの設計指示)**:
#
#   現状: `StageG` は Rectified Flow で残差 `r = 正解 - StageF出力` を生成する。
#   3,000 step 学習して目視した結果、テクスチャは実際に生成された
#   (grain 0.63 -> 1.27) 一方、**平坦な領域を悪化させた**。空・なだらかな階調・
#   暗部に**斑模様**が出る。黄色いボケの例では、薄い楕円だったものが濃い
#   オリーブ色の塊+黒点になった。
#
#   原因: 平坦な領域では正解の残差がほぼ 0 だが、フローは**全振幅のノイズから
#   積分を始める**ため、モデルが「ノイズを正確に打ち消す」ことを学ばない限り
#   斑が残る。学習量で消える保証がない。
#
#   処方: 残差を**局所的な期待振幅 `s` で正規化**して学習する。平坦な領域では
#   `s ≈ 0` なので、サンプリング後に `s` を掛け戻せば**出力も自動的に 0 になる**。
#   モデルは「相対的なテクスチャの形」だけを学べばよくなり、学習も楽になる。
#
#   効かない条件 (必ず検出できるようにする): `s` の推定に**穴の中の正解を
#   使ってはならない** (漏洩)。既知画素だけから推定すること。漏洩があると
#   学習時だけ性能が出て推論で崩れる。`tests/test_stage_g2.py` の「穴の中身だけ
#   を変えても `residual_scale` の出力が完全一致する」テストで担保する。
# ============================================================================


def _fill_scale_inward(val: torch.Tensor, valid: torch.Tensor, iters: int) -> torch.Tensor:
    """有効な (`valid==1`) 画素の値を、無効な画素へ外側から内側へ伝播させる。

    `stage_b_search._fill_blocks_inward` と同じ「縁から内側へ滲ませる」発想だが、
    離散インデックス (オフセット番号) ではなく連続値 (振幅) を伝えるため、最頻値
    近似の `max_pool2d` ではなく `stage_b_simple._propagate` と同じ近傍平均
    (`_box` による重み付き平均) を使う。連続量には平均のほうが自然で、境界での
    ジャンプも出にくい。

    `val`, `valid`: (N,1,H,W)。`valid==0` の画素は初期値を持たないものとして
    扱い (`val*valid` で 0 埋め)、`valid==1` の領域が近傍 (3x3) へ広がるたびに
    その近傍平均で埋めていく。`valid` が一度も 1 にならない画像 (全画素が穴で
    既知画素が皆無) は、伝播する種が無いため全域 0 のまま返る (有限値なので
    契約は満たすが、その場合はそもそも推定不能なことを示している)。
    """
    k = valid.to(val.dtype)
    v = val * k
    for _ in range(iters):
        if float((1.0 - k).sum()) == 0:
            break
        nb = _box(k, 1)
        grow = ((nb > 1e-6).to(k.dtype) - k).clamp(0.0, 1.0)
        if float(grow.sum()) == 0:
            break
        avg = _box(v, 1) / (nb + 1e-6)
        v = torch.where(grow > 0, avg, v)
        k = (k + grow).clamp(0.0, 1.0)
    return v


def residual_scale(img: torch.Tensor, mask: torch.Tensor, blur_sigma: float = 2.0,
                   box: int = 15, fill_iters: int = 64) -> torch.Tensor:
    """残差の期待振幅 `s` を**既知画素だけ**から推定する ((N,1,H,W) を返す)。

    このモジュール冒頭の「効く理屈」の核。手順:

      1. 高周波成分 `hp = img - blur(img)`。ぼかしは分離可能ガウシアン
         (`stage_b_search._gaussian_blur`、`blur_sigma`)。`F.interpolate(...,
         mode="area")` は MPS で落ちるため使わない (このモジュールの他の関数と
         同じ理由。実際には `_gaussian_blur` は conv2d ベースで interpolate 自体
         を使わない)。
      2. 既知画素 (`mask==0`) だけで `hp^2` の局所平均を取る:
         `num = box_filter(hp^2 * (1-mask))`, `den = box_filter(1-mask)`,
         `s2 = num / (den + eps)`。box は一辺 `box` の平均フィルタ
         (`stage_b_simple._box`、半径は `box // 2`)。チャンネルは平均して 1ch。
      3. 穴の内部は `den` が 0 に近く推定できないので、box 窓の過半数が既知
         画素で埋まっている画素だけを「種」とし、`_fill_scale_inward` で
         外側から内側へ伝播させる (`stage_b_search._fill_blocks_inward` と
         同じ発想)。
      4. `s = sqrt(s2)` を返す。マスク外も含め全画素で有限。

    **漏洩を断つ仕掛け (ここが最重要)**: 素の `_gaussian_blur(img, sigma)` を
    そのまま使うと、穴の縁の既知画素であっても、ぼかしカーネルが `sigma` に
    応じた範囲で穴の内部の画素を巻き込んでしまい、その既知画素の `hp` が
    わずかに「穴の中身」に依存してしまう。これは「穴の中の正解を使っては
    ならない」という制約に抵触する。そこで低周波成分は

        blurred = blur(img * known) / (blur(known) + eps)

    という**既知画素だけを使った正規化畳み込み**で求める。分子・分母とも
    `known` (= `mask<=0.5`) で重み付けしているので、`blurred` の値は穴の中身を
    どう変えても一切変わらない。さらに、高周波成分 `hp = img - blurred` の
    うち実際に使うのは `hp^2 * known` (穴の外だけ) なので、穴の中の `img` の
    値そのものは (`hp` の計算には使われても) 最終的な集計から完全に除外される。
    これにより `s` はパイプライン全体を通じて既知画素の情報だけから決まる。
    `tests/test_stage_g2.py` で「穴の中身だけを変えた2枚の画像で `s` が
    完全一致する」ことを直接検証している (漏洩があれば一致しないので即座に
    検出できる)。
    """
    known = (mask <= 0.5).to(img.dtype)                     # 1 = 既知画素

    num_blur = _gaussian_blur(img * known, blur_sigma)
    den_blur = _gaussian_blur(known, blur_sigma)
    blurred = num_blur / (den_blur + 1e-6)
    hp = img - blurred
    hp2 = hp.pow(2).mean(dim=1, keepdim=True)                # 3ch -> 1ch (チャンネル平均)

    r = max(box // 2, 1)
    num = _box(hp2 * known, r)
    den = _box(known, r)
    s2 = num / (den + 1e-6)

    # box窓の過半数が既知画素で埋まっている画素だけ、局所推定を信用する。
    # 埋まっていない (穴の奥に近い) 画素は _fill_scale_inward で外側から伝播させる。
    valid = (den > 0.5).to(img.dtype)
    s2 = _fill_scale_inward(s2, valid, fill_iters)

    return torch.sqrt(torch.clamp(s2, min=0.0))


def make_cond9(sf_out: torch.Tensor, mask: torch.Tensor, dist: torch.Tensor,
              copy_resid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """条件テンソル (N,9,H,W) を組み立てる — `make_cond` (8ch) に `scale` を1ch追加。

    **既存の `make_cond` は変更しない** (既存のテストと checkpoint を壊さない
    ため)。この関数は `make_cond` をそのまま呼び出し、末尾に `scale` を連結
    するだけで、8ch 分のチャンネル順ロジックを二重実装しない。

    **チャンネル順は固定**:
      ch 0-2 : `sf_out`      StageF出力 (知覚空間3ch)
      ch 3   : `mask`        1 = 穴
      ch 4   : `dist / 32`   穴の内部距離を 32 で割って [0,1] にクリップしたもの
      ch 5-7 : `copy_resid`  Stage B の検索パッチ残差 (copy - sf_out)
      ch 8   : `scale`       `residual_scale` の出力。局所的な残差振幅の推定値。
                              平坦な領域では 0 に近く、サンプリング後にこれを
                              掛け戻すことで出力も自動的に 0 に近づく (モジュール
                              冒頭の「効く理屈」参照)

    ch 3 の位置は 8ch のときと変わらないので、`rectified_flow_loss` /
    `rectified_flow_loss_scaled` はどちらの cond でも同じ読み方でマスクを取れる。
    """
    return torch.cat([make_cond(sf_out, mask, dist, copy_resid), scale], dim=1)


def rectified_flow_loss_scaled(model: StageG, r: torch.Tensor, cond: torch.Tensor,
                               scale: torch.Tensor, rng: torch.Generator | None = None):
    """`rectified_flow_loss` の正規化版。目標を `r_n = r / (scale + eps)` にする。

    正規化以外は `rectified_flow_loss` と完全に同じ (マスク重み付き L2、
    `t~U(0,1)`、速度 `v = r_n - noise`) なので、実装は既存の
    `rectified_flow_loss` をそのまま呼び出すだけにして二重実装を避ける。

    `eps` は `scale` の**全体中央値の 1e-3 倍程度**にする。固定の小定数
    (例えば 1e-6) だと、`scale` の絶対スケールが画像やチェックポイントで
    変わったときに「実質ゼロ除算」になる場所が出たり、逆に不要に大きく
    割ってしまったりする。中央値に対する相対値にすることでスケールに追従する。
    ただし `scale` が全域 0 (病的なケース、テストで作りうる) だと中央値も 0 に
    なり `eps` が 0 になってしまうので、最低限のフロア (1e-8) を必ず入れて
    **ゼロ除算で発散しないこと**を保証する。

    `r_n` は外れ値対策で `clamp(-8, 8)` する — 平坦な領域では `scale` が
    非常に小さくなるため、正規化するとわずかな残差でも `r_n` が極端に
    大きくなりうる。学習初期にこれが loss を吹き飛ばして崩壊させるのを防ぐ。
    """
    eps = max(float(scale.median()) * 1e-3, 1e-8)
    r_n = (r / (scale + eps)).clamp(-8.0, 8.0)
    return rectified_flow_loss(model, r_n, cond, rng=rng)


def sample_residual_scaled(model: StageG, cond: torch.Tensor, mask: torch.Tensor,
                           scale: torch.Tensor, steps: int = 8,
                           generator: torch.Generator | None = None) -> torch.Tensor:
    """`sample_residual` の正規化版。正規化空間でサンプリングし、最後に `scale` を掛け戻す。

    `sample_residual` がそのまま返す `r_n_hat` (正規化空間の残差の標本) に
    `scale` を掛けるだけ。二重実装を避けるため、オイラー法の積分ループ自体は
    既存の `sample_residual` に委譲する。

    **これが「効く理屈」の出口**: `scale` が 0 に近い (平坦な) 領域では、
    モデルがどんな `r_n_hat` を出しても `r_n_hat * scale ≈ 0` になる。
    `sample_residual` は既にマスク外を 0 にして返す契約なので、ここでは
    それに `scale` を掛けるだけでよく、改めてマスクを掛け直す必要はない
    (マスク外は既に 0 であり、有限な `scale` を掛けても 0 のまま)。
    """
    r_n_hat = sample_residual(model, cond, mask, steps=steps, generator=generator)
    return r_n_hat * scale


def main() -> int:
    """パラメータ数を確認するための簡易 CLI (`stage_f.py` の `main()` と同じ形)。"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--cond-ch", type=int, default=8)
    a = ap.parse_args()

    model = StageG(base=a.base, cond_ch=a.cond_ch)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"StageG(base={a.base}, cond_ch={a.cond_ch}) params: {n_par:,}  "
          f"(3M以下: {'OK' if n_par <= 3_000_000 else 'NG'})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
