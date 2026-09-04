# nagi_inpaint2

100 メガピクセル・リニア HDR の写真から、**不要な人物や物体を消す**ためのモジュール。
M1 MacBook Air (16GB, ファンレス) で動く。[ShadeWave](https://github.com/uniuyuni) から
外部モジュールとして呼ばれることを想定している。

## 何ができるか

マスクで指定した範囲を消し、**候補を 2 つ返す**。採否は人が決める。

実測 (実在の人物 60 件・ユーザ本人による絶対評価。`eval/RESULTS.md`):

| | 採用率 | 95%信頼区間 | 秒/箇所 |
|---|---|---|---|
| LaMa 単独 | 63.3% | [50.7%, 74.4%] | 0.63 |
| YOEO+L5 単独 | 75.0% | [62.8%, 84.2%] | 5.3 |
| **2 案** | **88.3%** | **[77.8%, 94.2%]** | **5.9** |

**2 案にすると +13.3 点で +0.63 秒。** 得意な場面が違うので、
どちらか一方を選ぶ根拠が無い (McNemar p=0.21)。

残る失敗の中身は **人混み 5 / ボケた人物 2**。構造の破綻は 0 件だった。

## 使い方

```python
from nagi_inpaint2 import api

pipe, device = api.setup(device="mps")          # 重みは約 7GB。使い回すこと

# 1 案 (ShadeWave の既存契約と同じ形。in-place 更新して同じ配列を返す)
image = api.predict_helper(pipe, image, mask, bbox)

# 2 案 (画像は書き換えない。位置とパッチだけ返る)
for name, (y, x, patch) in api.predict_candidates(pipe, image, mask, bbox).items():
    out = image.copy()
    out[y:y + patch.shape[0], x:x + patch.shape[1], :3] = patch
```

- `image` : `(H, W, 3)` float32 RGB **リニア**。**1.0 超 (HDR) 可**。書き込み可能なこと
- `mask`  : `(H, W)` または `(H, W, 1)` float32。**1 が除去対象**。画像全体のマスク
- `bbox`  : `(x, y, w, h)`。**y,x ではない。** 塗った箇所ごとに 1 回ずつ呼ぶ

**保証**

- **マスクの外は 1 ビットも変わらない** (`tests/test_api.py` で担保)
- **費用はマスクの面積だけに依存する。** 1 億画素でもクロップ 1 枚分で済む
- 同じ入力なら同じ出力 (シード固定)

## 重み

`models/` に置く。`$NAGI_INPAINT2_MODELS` で場所を変えられる。**リポジトリには含まれない。**

| 置き場所 | 中身 | ライセンス |
|---|---|---|
| `models/big-lama.pt` | LaMa (196MB) | Apache-2.0 |
| `models/yoeo/yoeo.pth` | YOEO の UNet 重み (3.2GB) | Apache-2.0 |
| `models/sd15-inpaint/` | SD1.5-inpainting の VAE / text encoder / scheduler | CreativeML OpenRAIL-M |

```bash
curl -fL -o models/big-lama.pt \
  https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```

YOEO と SD1.5-inpainting の取得手順は `docs/DESIGN_V2_JA.md` §7 を参照。

## 構成

```
nagi_inpaint2/     製品経路。api -> pipeline -> removers -> lama / stage_d
  api.py             ShadeWave 契約 (setup / predict_helper / predict_candidates)
  pipeline.py        L1〜L5 の組み立て。MultiInpainter が候補を複数返す
  removers.py        除去コアの薄い包み (LaMa / YOEO / SmartEraser / RORem)
  stage_d.py         L5 Poisson 合成 (継ぎ目を消す)
  perceptual_space.py  リニア <-> 知覚空間。torch も libraw も引かない
  paths.py           重みの場所を作業ディレクトリに依存させず解決する
eval/              評価基盤。盲検A/B と実測結果 (RESULTS.md)
data/              ベンチの作成 (実在の人物を検出して切り出す)
tests/             53 件。モデルを読まないので数秒で回る
docs/              設計と、撤回した主張の記録
```

## 開発

```bash
pixi run python -m pytest tests/ -q          # 53 件 / 約 7 秒
pixi run python eval/verify_100mp.py         # 1 億画素での通し確認
```

## 経緯

**v1 は盲検A/B で 0勝16敗だった。** 原因は代理指標 (簡易 LPIPS) で判断したことと、
点推定の損失が多峰な目標に対して平均へ回帰することだった。
v2 は既存の除去モデルを組み合わせ、**合否は人手の盲検A/Bだけで決める**方針に変えた。

途中で否定した仮説と、その根拠となる実測は `eval/RESULTS.md` に全部残してある
(蒸留・Real-ESRGAN・視野の拡大・RORem・自動検品など)。
**撤回した主張は消さずに、撤回した理由を添えて残している。**
