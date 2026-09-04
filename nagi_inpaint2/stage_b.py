"""Stage B — 学習版の密コストボリューム合成 (SPEC §4.2)。

非学習プロトタイプ (`stage_b_simple.py`) は LaMa の失敗条件でも負け
(PSNR 25.43 vs 29.53)、目視で**くしゃくしゃに潰れた artifact** が残った。
診断された主因は「argmin で単一候補を貼っている」ことと「特徴が固定」であること。

**初版の学習実装も失敗した。** 実測で2つの欠陥が出た:

  1. マスク内画素の **91%** が有効候補を1つも持てなかった
     (塊状マスクの内部では参照パッチが丸ごと穴に入り、全候補が無効になる)
  2. soft ブレンドが τ によらず**均等平均に退化**した
     (エントロピー 1.000、1位重み 0.25)。初期オフセットの ±3 局所窓から
     取った 49候補は**どれもほぼ同じ場所**を指しており、コスト差が 4e-6 しかない。
     ほぼ同一のパッチを4枚平均する = わずかにぼかすだけ

**設計上の含意**: top-k soft ブレンドは**粗レベルの広域探索から得た候補**に対して
行わなければ意味がない。局所窓の top-k は「ほぼ同じ画素の k 個のコピー」である。
そして穴の内部画素は、原理的に粗レベル (将来は Stage A) からしか
オフセットを得られない。Stage A は「あれば良い」ではなく**必須**だった。

**本モジュールの構成**

  1. 粗レベル (穴が数画素まで縮み文脈がある) で広域探索し、
     **NMS で互いに離れた上位 k 個**のオフセットとコストを保持する
  2. 各候補を独立に細レベルへ伝播し、局所小窓で精緻化する
  3. 粗レベルのコスト (常に有効) を土台に、細レベルで有効な分を足して
     **softmax でブレンド**する

コストは画素ごとに最小値を引いて正規化するので、τ は「効いてくるコスト差」として
解釈できる。コストボリューム自体は P2b で MPS 実効 130M eval/s を確認済み。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage_b_simple import MIN_W, _box, _features as _handcrafted, _fill_inward, _shift


class FeatureNet(nn.Module):
    """照合用の特徴抽出器。**手作り特徴を土台に、学習分を残差として足す。**

    学習特徴だけで照合すると、ランダム初期化の CNN 出力はほぼノイズで、
    非学習プロトタイプの「生RGB + 勾配」に遥かに劣る地点から始まる。
    実測で検証 PSNR 15.8-17.6 dB (プロトタイプ 25.4 dB) と大きく負けた。

    そこで手作り特徴 (5ch: RGB + 勾配x + 勾配y) を常に含め、
    学習枝は**最終層をゼロ初期化**して no-op から始める。
    こうすれば初期状態でプロトタイプと同等になり、学習は改善しかもたらさない。

    実行時間はコストボリュームが支配的 (特徴抽出は1回、コストボリュームは K 回)
    なので、チャンネルを増やしてもほぼ無料。
    """

    def __init__(self, cin: int = 4, cmid: int = 64, cout: int = 8, depth: int = 4):
        super().__init__()
        layers = [nn.Conv2d(cin, cmid, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(cmid, cmid, 3, padding=1), nn.GELU()]
        last = nn.Conv2d(cmid, cout, 3, padding=1)
        # **ゼロ初期化してはならない。** コストは (f - fs)^2 の形で、
        # クエリ側 f と参照側 fs は同じ FeatureNet を通る。学習枝をゼロにすると
        # 両側とも 0 になり、差も 0、その勾配 2(f-fs) も**厳密に 0** になる。
        # 実測で 20,000 step / 27時間、パラメータが1ビットも動かなかった。
        # 小さいが非ゼロにすることで、初期性能をほぼ保ちつつ勾配を流す。
        nn.init.normal_(last.weight, std=1e-2)
        nn.init.zeros_(last.bias)
        layers += [last]
        self.net = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base = _handcrafted(img)                       # (B,5,H,W) 手作り
        learned = self.net(torch.cat([img, mask], 1))  # (B,8,H,W) 初期はゼロ
        return torch.cat([base, learned], 1) * (1.0 - mask)


def _cost_volume(f, valid, r, radius, stride=1):
    """全オフセットのコストを返す。(B, K, H, W) と オフセット一覧。"""
    costs, offs = [], []
    for oy in range(-radius, radius + 1, stride):
        for ox in range(-radius, radius + 1, stride):
            if oy == 0 and ox == 0:
                continue
            fs, vs = _shift(f, oy, ox), _shift(valid, oy, ox)
            w = valid * vs
            wb = _box(w, r)
            c = _box(((f - fs) ** 2).sum(1, keepdim=True) * w, r) / (wb + 1e-6)
            costs.append(torch.where((vs <= 0.5) | (wb < MIN_W),
                                     torch.full_like(c, float("inf")), c))
            offs.append((oy, ox))
    return torch.cat(costs, 1), offs


def _nms_topk(C, offs, k: int, nms_radius: int = 3):
    """互いに離れた上位 k 候補を選ぶ。

    素の topk は隣接オフセットが並ぶだけで「ほぼ同じ場所の k 個のコピー」になり、
    ブレンドが均等平均に退化する (初版の失敗)。
    最良を採るたびに、その周囲のオフセットを候補から外す。
    """
    B, K, H, W = C.shape
    dev = C.device
    oy_t = torch.tensor([o[0] for o in offs], device=dev, dtype=C.dtype)
    ox_t = torch.tensor([o[1] for o in offs], device=dev, dtype=C.dtype)
    work = C.clone()
    sel_oy, sel_ox, sel_c = [], [], []
    for _ in range(k):
        c, i = work.min(dim=1)                       # (B,H,W)
        sel_c.append(c)
        sel_oy.append(oy_t[i])
        sel_ox.append(ox_t[i])
        d = ((oy_t[None, :, None, None] - oy_t[i][:, None]) ** 2
             + (ox_t[None, :, None, None] - ox_t[i][:, None]) ** 2)
        work = torch.where(d <= nms_radius ** 2, torch.full_like(work, float("inf")), work)
    return (torch.stack(sel_oy, 1), torch.stack(sel_ox, 1), torch.stack(sel_c, 1))


@torch.no_grad()
def _refine_candidates(im, mk, feat, oy, ox, cost, radius: int, patch: int):
    """このレベルで k 個の候補オフセットをそれぞれ局所精緻化する。"""
    r = patch // 2
    valid = 1.0 - mk
    f = feat(im, mk)
    _, _, h, w = im.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=im.device, dtype=im.dtype),
        torch.arange(w, device=im.device, dtype=im.dtype), indexing="ij")

    def warp(x, dy, dx, mode="bilinear"):
        sy = (yy[None, None] + dy).clamp(0, h - 1)
        sx = (xx[None, None] + dx).clamp(0, w - 1)
        g = torch.stack([sx / (w - 1) * 2 - 1, sy / (h - 1) * 2 - 1], -1)[:, 0]
        return F.grid_sample(x, g, mode=mode, align_corners=True)

    for j in range(oy.shape[1]):
        f_w = warp(f, oy[:, j:j + 1], ox[:, j:j + 1])
        v_w = warp(valid, oy[:, j:j + 1], ox[:, j:j + 1], mode="nearest")
        best = torch.full_like(mk, float("inf"))
        bdy = torch.zeros_like(mk)
        bdx = torch.zeros_like(mk)
        for ddy in range(-radius, radius + 1):
            for ddx in range(-radius, radius + 1):
                fs, vs = _shift(f_w, ddy, ddx), _shift(v_w, ddy, ddx)
                wgt = valid * vs
                wb = _box(wgt, r)
                c = _box(((f - fs) ** 2).sum(1, keepdim=True) * wgt, r) / (wb + 1e-6)
                c = torch.where((vs <= 0.5) | (wb < MIN_W),
                                torch.full_like(c, float("inf")), c)
                upd = c < best
                best = torch.where(upd, c, best)
                bdy = torch.where(upd, torch.full_like(bdy, float(ddy)), bdy)
                bdx = torch.where(upd, torch.full_like(bdx, float(ddx)), bdx)
        ok = torch.isfinite(best)
        oy[:, j:j + 1] = oy[:, j:j + 1] + torch.where(ok, bdy, torch.zeros_like(bdy))
        ox[:, j:j + 1] = ox[:, j:j + 1] + torch.where(ok, bdx, torch.zeros_like(bdx))
        cost[:, j:j + 1] = torch.where(ok, best, cost[:, j:j + 1])
        a, b = _fill_inward(oy[:, j:j + 1], ox[:, j:j + 1], ok.to(mk.dtype), mk)
        oy[:, j:j + 1], ox[:, j:j + 1] = a, b
    return oy, ox, cost


@torch.no_grad()
def init_offsets_topk(img, mask, feat, levels: int = 5, coarse_radius: int = 16,
                      patch: int = 7, k: int = 4, nms_radius: int = 3,
                      refine_radius: int = 2):
    """粗レベルで広域探索し、**各レベルで精緻化しながら**全解像度へ降ろす。

    **各レベルでの精緻化を省いてはならない。** 初版は最粗レベルで解いた後
    一気に ×16 拡大し、全解像度で ±2 の局所探索を1回するだけだった。
    32×32 で解いたオフセットを ×16 すると誤差は最大 ±16px あり、
    ±2 の窓では直せない。その結果、学習版が非学習プロトタイプより
    **9 dB 悪い** (16.39 vs 25.43 dB) という結果になり、
    20,000 step / 10.7時間を「多段精緻化を捨てるとどれだけ悪化するか」の
    測定に費やしてしまった。

    戻り値: oy, ox, cost — いずれも (B, k, H, W)。
    Stage A が完成すればこの関数が Stage A の出力に置き換わる。
    """
    imgs, masks = [img], [mask]
    for _ in range(levels - 1):
        if min(imgs[-1].shape[-2:]) < 48:
            break
        imgs.append(F.avg_pool2d(imgs[-1], 2))
        masks.append((F.avg_pool2d(masks[-1], 2) > 0.05).to(mask.dtype))

    im, mk = imgs[-1], masks[-1]
    C, offs = _cost_volume(feat(im, mk), 1.0 - mk, patch // 2, coarse_radius)
    oy, ox, cost = _nms_topk(C, offs, k, nms_radius)

    # 粗レベルでも文脈が足りなかった画素は、縁から伝播させて埋める。
    # **コストも一緒に伝播させる。** 定数で埋めると候補間に差が無くなり、
    # softmax が均等平均に退化して soft ブレンドの意味が消える (初版の失敗)。
    solved = torch.isfinite(cost[:, :1]).to(mk.dtype)
    big = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost))
    for j in range(k):
        a, b = _fill_inward(oy[:, j:j + 1], ox[:, j:j + 1], solved, mk)
        oy[:, j:j + 1], ox[:, j:j + 1] = a, b
        c1, _ = _fill_inward(big[:, j:j + 1], big[:, j:j + 1], solved, mk)
        big[:, j:j + 1] = c1
    cost = big

    # 各レベルで精緻化しながら降ろす
    for l in range(len(imgs) - 2, -1, -1):
        h, w = imgs[l].shape[-2:]
        up = lambda t: F.interpolate(t, (h, w), mode="nearest")
        oy, ox, cost = up(oy) * 2, up(ox) * 2, up(cost)
        oy, ox, cost = _refine_candidates(imgs[l], masks[l], feat, oy, ox, cost,
                                          refine_radius, patch)
    return oy, ox, cost


class StageB(nn.Module):
    """候補ごとの局所精緻化と top-k soft 合成。"""

    def __init__(self, radius: int = 2, patch: int = 7, topk: int = 4,
                 tau: float = 0.05):
        super().__init__()
        self.f = FeatureNet()
        self.radius, self.patch, self.topk = radius, patch, topk
        # **τ は学習させない。** 学習させると L1 損失が τ を増やす方向へ動かす:
        # k 個の候補を平均すれば L1 は必ず下がるので、損失は「実画素を1枚選ぶ」より
        # 「k 枚ぼかして混ぜる」を報酬する。実測で τ 0.53 -> 1.21、
        # 重みエントロピー 0.71 -> 0.87 と、設計意図と逆向きに 17,500 step 進んだ。
        # そしてそのぼけこそ、我々が LaMa の失敗として批判していたものである。
        #
        # soft ブレンドは**学習を微分可能にするための手段**であって、
        # 推論時は実パッチを1枚選ぶ hard selection が本来の姿。
        # よって τ は外から与え、学習中はスケジュールで焼き鈍す
        # (differentiable-argmin 系の標準的な扱い)。
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward_debug(self, img, mask, oy_k, ox_k, cost_k, tau=None):
        """コストと候補も返す。候補ランキング損失 (probes/p8) 用。"""
        self._debug = True
        comp, conf = self.forward(img, mask, oy_k, ox_k, cost_k, tau)
        self._debug = False
        C, V = self._last
        return comp, conf, C, V

    def _warp(self, x, oy, ox, mode="bilinear"):
        _, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
        sy = (yy[None, None] + oy).clamp(0, h - 1)
        sx = (xx[None, None] + ox).clamp(0, w - 1)
        g = torch.stack([sx / (w - 1) * 2 - 1, sy / (h - 1) * 2 - 1], -1)[:, 0]
        return F.grid_sample(x, g, mode=mode, align_corners=True)

    def forward(self, img, mask, oy_k, ox_k, cost_k, tau: float | None = None,
                query=None):
        """oy_k / ox_k / cost_k は (B, k, H, W)。粗レベル由来の候補。

        query : Stage A の低周波補完 L で穴を埋めた画像 (B,3,H,W)。
                与えると穴の内部にも照合の手がかりが生まれ、
                **クエリ側の有効マスクが全画素 1 になる**ため、
                塊状マスク内部の 91% が評価不能だった問題が解消する (§4.1)。

        **クエリ側とソース側で有効マスクを分ける。**
          クエリ側: L で埋めた画素も「中身がある」ので有効
          ソース側: 穴の中から画素をコピーしてはならないので無効のまま
        この区別を怠ると、穴の中の (L で埋めた) 画素を参照元に選んでしまう。

        戻り値: (合成結果, 信頼度)。合成結果はマスク外にも伸びる (§4.4 の契約)。
        """
        r = self.patch // 2
        src_valid = 1.0 - mask                       # コピー元として使ってよいか
        if query is None:
            valid = src_valid                        # クエリ側も穴を除く (従来)
            f = self.f(img, mask)
        else:
            valid = torch.ones_like(mask)            # クエリ側は全画素有効
            f = self.f(query, torch.zeros_like(mask))

        # warp は勾配経路に置かない。torch 2.12 の MPS には
        # grid_sampler_2d_backward が無い。候補画素そのものは学習対象ではなく、
        # 学ぶのは「どれを選ぶか」= コスト経由の重みだけなので、
        # warp 済み画像を detach し特徴を取り直せば勾配は FeatureNet を正しく通る。
        vals, costs = [], []
        for j in range(oy_k.shape[1]):
            oy, ox = oy_k[:, j:j + 1], ox_k[:, j:j + 1]
            with torch.no_grad():
                i_w = self._warp(img, oy, ox)
                v_w = self._warp(src_valid, oy, ox, mode="nearest")
            i_w, v_w = i_w.detach(), v_w.detach()
            f_w = self.f(i_w, 1.0 - v_w)

            # 候補ごとにゼロ近傍の小窓で精緻化する。
            # **内側の探索は no_grad。** これは離散的な argmin で元々微分不可能であり、
            # k×(2r+1)² 回ぶんの中間テンソルを autograd グラフに載せると
            # 512²/batch4 で 18GB を超えて OOM する。
            # 勾配が要るのは最終的に選ばれた k 個のコストだけ。
            with torch.no_grad():
                best_c = torch.full_like(mask, float("inf"))
                best_d = torch.zeros(2, dtype=torch.long)
                bdy = torch.zeros_like(mask)
                bdx = torch.zeros_like(mask)
                for ddy in range(-self.radius, self.radius + 1):
                    for ddx in range(-self.radius, self.radius + 1):
                        fs, vs = _shift(f_w, ddy, ddx), _shift(v_w, ddy, ddx)
                        w = valid * vs
                        wb = _box(w, r)
                        c = _box(((f - fs) ** 2).sum(1, keepdim=True) * w, r) / (wb + 1e-6)
                        c = torch.where((vs <= 0.5) | (wb < MIN_W),
                                        torch.full_like(c, float("inf")), c)
                        upd = c < best_c
                        best_c = torch.where(upd, c, best_c)
                        bdy = torch.where(upd, torch.full_like(bdy, float(ddy)), bdy)
                        bdx = torch.where(upd, torch.full_like(bdx, float(ddx)), bdx)
                # 選ばれた精緻化オフセットで候補画素を引き直す
                best_v = self._warp(i_w, bdy, bdx)
                fine_ok = torch.isfinite(best_c)

            # 勾配付きで k 個ぶんだけコストを取り直す。
            # **参照枝 (fs) は detach する。** grid_sample の逆伝播が MPS に無いのと、
            # 学習信号はクエリ枝から流せば足りる (metric learning の stop-gradient と同じ)。
            with torch.no_grad():
                fs = self._warp(f_w, bdy, bdx).detach()
                vs = self._warp(v_w, bdy, bdx, mode="nearest").detach()
            w = valid * vs
            wb = _box(w, r)
            cg = _box(((f - fs) ** 2).sum(1, keepdim=True) * w, r) / (wb + 1e-6)
            # 細レベルで文脈が無い画素 (穴の内部) は粗レベルの伝播コストを使う
            costs.append(torch.where(fine_ok, cg, cost_k[:, j:j + 1]))
            vals.append(best_v.detach())

        C = torch.cat(costs, 1)                     # (B,k,H,W)
        V = torch.stack(vals, 2)                    # (B,3,k,H,W)

        # 画素ごとに最小値を引き、**さらに候補間のばらつきで割る**。
        # 引くだけではコストの絶対スケール (実測 1e-3、候補間の差 2e-5) に
        # τ を合わせ込まねばならず、初期値が 1000倍ずれて softmax が
        # 均等平均に張り付いたまま動かなかった。
        # 相対差にすれば τ は「何倍の差から効かせるか」というスケール不変な量になる。
        C = C - C.min(dim=1, keepdim=True).values
        C = C / (C.mean(dim=1, keepdim=True) + 1e-8)
        t = float(self.tau) if tau is None else tau
        wgt = torch.softmax(-C / max(t, 1e-3), dim=1)
        comp = (V * wgt[:, None]).sum(2)

        # 信頼度: 1位と2位のコスト差 (大きいほど確信がある)
        margin = C.topk(2, dim=1, largest=False).values[:, 1:2] if C.shape[1] > 1 \
            else torch.ones_like(C[:, :1])
        conf = torch.tanh(margin).clamp(0, 1)
        if getattr(self, "_debug", False):
            self._last = (C, V)
        # **マスク外でクリップして返さない** (§4.4 の契約)。
        # 勾配ドメイン合成は境界での元画像との食い違いを境界条件に使うため、
        # ここで元画像に戻すと食い違いが恒等的に 0 になり Stage D が無効化される。
        # この誤りは stage_d / stage_b_simple / stage_b で3度踏んだ。
        # マスク外を戻すのは呼び出し側の責任。
        return comp, conf
