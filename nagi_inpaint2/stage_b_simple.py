"""Stage B の非学習プロトタイプ (SPEC §4.2)。

学習済み特徴抽出器 `f` の代わりに生画素と勾配を使う、Stage B の骨格。
**本プロジェクトの中心的な主張を、学習なしで検証するためのもの**:

    「1億画素の実写では、穴を埋めるのに最適な画素はたいてい同じ画像の中にある。
      生成に頼るのは、手元の実データを捨てて劣化コピーを作る行為である」(§3.1)

実測で LaMa の失敗条件は **高周波テクスチャ × 塊状マスク** と判明した
(§1.3 の追記、blob で −7.04 dB)。葉・花・草・彫刻は
「似た画素が同じ画像に大量にある」典型なので、ここで実画素コピーが勝てるはず。

**オフセットの規約 (ここを間違えると正反対の位置から貼る)**

    o = source − query          out(q) = img(q + o(q))

初版はスコアリングが `roll(f, +d)` すなわち source = q − d を評価しながら、
貼り付けは q + d から行っていた。符号が逆で、鏡像位置の画素を貼っていた。

学習版との違い:
  - 特徴 `f` が固定 (生画素 + 勾配) で、露出差やスケール差への頑健性がない
  - ブレンド重みが学習されていない (上位1候補をそのまま使う)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _features(x: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) -> (1,C,H,W)。生画素に勾配を足しただけの固定特徴。"""
    g = x.mean(1, keepdim=True)
    gx = F.pad(g[:, :, :, 1:] - g[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(g[:, :, 1:, :] - g[:, :, :-1, :], (0, 0, 0, 1))
    return torch.cat([x, gx * 2.0, gy * 2.0], 1)


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """分離可能な箱フィルタ。パッチ和を O(1)/画素で得る。"""
    k = 2 * r + 1
    x = F.avg_pool2d(F.pad(x, (r, r, 0, 0), mode="replicate"), (1, k), stride=1)
    x = F.avg_pool2d(F.pad(x, (0, 0, r, r), mode="replicate"), (k, 1), stride=1)
    return x


def _shift(x: torch.Tensor, oy: int, ox: int) -> torch.Tensor:
    """out(q) = x(q + o)。範囲外は 0 で埋める (回り込ませない)。"""
    out = torch.roll(x, shifts=(-oy, -ox), dims=(2, 3))
    if oy > 0:
        out[:, :, -oy:, :] = 0
    elif oy < 0:
        out[:, :, :-oy, :] = 0
    if ox > 0:
        out[:, :, :, -ox:] = 0
    elif ox < 0:
        out[:, :, :, :-ox] = 0
    return out


MIN_W = 0.15      # パッチ内の既知画素の割合の下限


def _cost(f, valid, fs, vs, r):
    """既知画素同士だけでパッチ距離を取る。

    **重み和が小さい画素は inf にする。** これをしないと、パッチが丸ごと穴に入る
    穴の内部で 0/ε = 0 となり、「最初に評価したオフセット」が必ず選ばれてしまう
    (実測で探索窓の隅 oy=ox=-20 に張り付いた)。
    無効にしておけば、その画素は _fill_inward で縁から伝播した値を保つ。
    """
    w = valid * vs
    d2 = ((f - fs) ** 2).sum(1, keepdim=True) * w
    wb = _box(w, r)
    c = _box(d2, r) / (wb + 1e-6)
    bad = (vs <= 0.5) | (wb < MIN_W)
    return torch.where(bad, torch.full_like(c, float("inf")), c)


def _fill_inward(oy, ox, solved, mask, iters: int = 64):
    """穴の縁で決まったオフセットを内部へ伝播させる。

    内部画素は文脈が無いので自力ではオフセットを決められない。
    縁から内向きに滲ませることで、穴全体が一貫した参照元を持つ。
    """
    k = solved.clone()
    for _ in range(iters):
        if float((mask * (1 - k)).sum()) == 0:
            break
        nb = _box(k, 1)
        grow = (((nb > 1e-6).to(k.dtype) - k) * mask).clamp(0, 1)
        if float(grow.sum()) == 0:
            break
        # **整数に丸めてから伝播させる。** 平均したままだと小数オフセットになり、
        # grid_sample の補間でテクスチャが潰れて「くしゃくしゃの artifact」になる。
        # コピーに使うオフセットは整数でなければならない。
        oy = oy * (1 - grow) + torch.round(_box(oy * k, 1) / (nb + 1e-6)) * grow
        ox = ox * (1 - grow) + torch.round(_box(ox * k, 1) / (nb + 1e-6)) * grow
        k = (k + grow).clamp(0, 1)
    return oy, ox


def solve_offsets(img, mask, radius: int = 24, stride: int = 1, patch: int = 7):
    """全画素について最良の o = source − query を求める (密コストボリューム)。"""
    valid = 1.0 - mask
    f = _features(img) * valid
    r = patch // 2
    best = torch.full_like(mask, float("inf"))
    boy = torch.zeros_like(mask)
    box_ = torch.zeros_like(mask)
    for oy in range(-radius, radius + 1, stride):
        for ox in range(-radius, radius + 1, stride):
            if oy == 0 and ox == 0:
                continue
            c = _cost(f, valid, _shift(f, oy, ox), _shift(valid, oy, ox), r)
            upd = c < best
            best = torch.where(upd, c, best)
            boy = torch.where(upd, torch.full_like(boy, float(oy)), boy)
            box_ = torch.where(upd, torch.full_like(box_, float(ox)), box_)
    solved = torch.isfinite(best).to(mask.dtype)
    boy, box_ = _fill_inward(boy, box_, solved, mask)
    return boy, box_, best


def _propagate(o, mask, steps: int):
    """穴の中で解いたオフセットを、その周囲の帯へにじみ出させる。

    warp してからパッチ比較する方式では、パッチが穴の外にはみ出す部分でも
    同じオフセットで参照していないと比較が不整合になる。
    """
    known = mask.clone()
    for _ in range(steps):
        nb = _box(known, 1)
        avg = _box(o * known, 1) / (nb + 1e-6)
        grow = ((nb > 1e-6).to(o.dtype) - known).clamp(0, 1)
        o = o * (1 - grow) + avg * grow
        known = (known + grow).clamp(0, 1)
    return o


def _refine(img, mask, oy, ox, radius: int, patch: int):
    """warp してからゼロ近傍の固定小窓で探索する (PWC-Net 方式, §4.2 step 2-3)。

    画素ごとに異なるオフセットで直接 `grid_sample` するとパッチ内の参照先が
    ばらばらになり比較が壊れる。先に**特徴マップ全体を warp**してしまえば、
    残差の探索は固定シフトで済み、パッチ内で一貫する。
    """
    _, _, h, w = img.shape
    r = patch // 2
    valid = 1.0 - mask
    f = _features(img) * valid

    # 帯へオフセットを伝播させてから warp する
    poy, pox = _propagate(oy, mask, patch), _propagate(ox, mask, patch)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype), indexing="ij")
    sy = (yy[None, None] + poy).clamp(0, h - 1)
    sx = (xx[None, None] + pox).clamp(0, w - 1)
    g = torch.stack([sx / (w - 1) * 2 - 1, sy / (h - 1) * 2 - 1], -1)[:, 0]
    f_w = F.grid_sample(f, g, mode="bilinear", align_corners=True)
    v_w = F.grid_sample(valid, g, mode="nearest", align_corners=True)

    best = torch.full_like(mask, float("inf"))
    boy, box_ = oy.clone(), ox.clone()
    for ddy in range(-radius, radius + 1):
        for ddx in range(-radius, radius + 1):
            c = _cost(f, valid, _shift(f_w, ddy, ddx), _shift(v_w, ddy, ddx), r)
            upd = c < best
            best = torch.where(upd, c, best)
            boy = torch.where(upd, oy + ddy, boy)
            box_ = torch.where(upd, ox + ddx, box_)
    # 精緻化できなかった内部画素は、縁で更新された値を引き継ぐ
    solved = torch.isfinite(best).to(mask.dtype)
    return _fill_inward(boy, box_, solved, mask)


def fill_pyramid(img: torch.Tensor, mask: torch.Tensor, levels: int = 5,
                 coarse_radius: int = 16, refine_radius: int = 2,
                 patch: int = 7) -> torch.Tensor:
    """多段で対応場を解く。**大きな穴にはこれが必須。**

    単一スケールだと、穴の内部画素の参照パッチが丸ごと穴に入ってしまい
    重みが全ゼロ = 全オフセットのコストが 0 になって argmin が無意味になる。
    粗い階層では穴が数画素まで縮み、パッチが穴の外に届くので文脈が得られる。

    **返り値はマスク外にも合成結果が伸びている** (Stage D の契約, §4.4)。
    マスク外を元画像に戻すのは呼び出し側の責任。
    """
    imgs, masks = [img], [mask]
    for _ in range(levels - 1):
        if min(imgs[-1].shape[-2:]) < 48:
            break
        imgs.append(F.avg_pool2d(imgs[-1], 2))
        masks.append((F.avg_pool2d(masks[-1], 2) > 0.05).to(mask.dtype))

    oy, ox, _ = solve_offsets(imgs[-1], masks[-1], coarse_radius, 1, patch)
    for l in range(len(imgs) - 2, -1, -1):
        h, w = imgs[l].shape[-2:]
        oy = F.interpolate(oy, (h, w), mode="nearest") * 2
        ox = F.interpolate(ox, (h, w), mode="nearest") * 2
        oy, ox = _refine(imgs[l], masks[l], oy, ox, refine_radius, patch)

    _, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype), indexing="ij")
    sy = (yy[None, None] + oy).clamp(0, h - 1)
    sx = (xx[None, None] + ox).clamp(0, w - 1)
    g = torch.stack([sx / (w - 1) * 2 - 1, sy / (h - 1) * 2 - 1], -1)[:, 0]
    return F.grid_sample(img, g, mode="bilinear",
                         padding_mode="reflection", align_corners=True)


def inpaint_np(p: np.ndarray, mask: np.ndarray, device: str = "mps",
               levels: int = 5, coarse_radius: int = 16,
               refine_radius: int = 2, patch: int = 7) -> np.ndarray:
    """numpy (H,W,3) 知覚空間 + (H,W) マスク -> numpy (マスク外にも伸びた合成結果)。"""
    x = torch.from_numpy(p.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    m = torch.from_numpy(mask[None, None].astype(np.float32)).to(device)
    with torch.inference_mode():
        y = fill_pyramid(x, m, levels, coarse_radius, refine_radius, patch)
    return y[0].permute(1, 2, 0).float().cpu().numpy()
