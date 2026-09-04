"""既存の除去モデルを、本リポジトリの評価基盤に差し込める形で包む (v2 Phase 0)。

**インタフェースは `eval/lama_baseline.Lama` に合わせる。**
`remover(p_hwc, mask_hw) -> p_hwc` (知覚空間で受けて知覚空間で返す)。
こうしておくと `eval/benchmark.py` / `eval/ab_compare.py` / `eval/quickval.py` が
そのまま使える。**評価コードを書き直さない**のは SPEC R11 の要求でもある。

---

## YOEO (CVPR'26) — `models/yoeo/yoeo.pth`

**重みの正体 (実測で確認済み)**

`yoeo.pth` は diffusers の `EMAModel` の state で、`shadow_params` に
**686 個のテンソルが名前無しの順序付きリスト**で入っている。
公式の SD1.5-inpainting UNet を config から作って `parameters()` を並べたところ、
**686 個の形状が完全一致**した (先頭 `(320, 9, 3, 3)` = 9ch 入力、
末尾 `(4, 320, 3, 3)`、計 859.5M params)。よって順序で載せてよい。

**したがって YOEO 単体では動かない。** VAE・テキストエンコーダ・スケジューラは
`stable-diffusion-v1-5/stable-diffusion-inpainting` (CreativeML OpenRAIL-M) から取る。
土台の UNet 重みは YOEO が置き換えるので取得不要 (3.4GB 節約)。

**推論レシピ (本家 `unhcv/projects/rlhf/test.py` から書き写した)**

  - スケジューラ: `CustomLCMScheduler` = `LCMScheduler` のサブクラスで
    タイムステップを直書き。**2 step なら `[999, 749]`**
  - `guidance_scale = 1` (CFG 無し)
  - `encoder_hidden_states = text_encoder([""])` — **空プロンプト**
  - 画像は `/127.5 - 1` で [-1,1]、マスクは `/255` で [0,1]
  - `masked_image_latent = vae.encode(image * (1 - mask)) * scaling_factor`
  - マスクは潜在の解像度へ `mode="nearest-exact"` で縮小
  - UNet 入力は `cat([latent(4), mask(1), masked_latent(4)], dim=1)` の 9ch

**決定性について**: 本家は VAE encode に `latent_dist.sample()` を使うが、
SPEC §9.1 は「同一入力 → ビット一致」を要求している。既定では `.mode()` を使い、
`vae_sample=True` で本家に合わせられるようにしてある。初期ノイズもシード固定。
"""
from __future__ import annotations

import os

import logging

import numpy as np
import torch
import torch.nn.functional as F

from nagi_inpaint2.paths import resolve

# diffusers の LCMScheduler は「749 は元の蒸留スケジュールに無い」と警告するが、
# 本家 YOEO の CustomLCMScheduler は**その検証ロジックを全てコメントアウトして**
# timesteps_dict を直接代入しているだけである (unhcv/third_party/distill/utils.py)。
# 代入後の状態 (timesteps / custom_timesteps / num_inference_steps / step index) が
# 一致することを確認済みなので、表示だけの問題として黙らせる。
logging.getLogger("diffusers.schedulers.scheduling_lcm").setLevel(logging.ERROR)

SD15_INPAINT = "models/sd15-inpaint"
YOEO_CKPT = "models/yoeo/yoeo.pth"

# CustomLCMScheduler が直書きしているタイムステップ表
YOEO_TIMESTEPS = {1: [999], 2: [999, 749], 3: [999, 749, 499], 4: [999, 749, 499, 249]}


class Yoeo:
    """YOEO による物体除去。`Lama` と同じ呼び出し方ができる。

    `p` (H,W,3 知覚空間) と `mask` (H,W) を受け、**マスク内を埋めた** 知覚空間を返す。
    正規化の流儀は呼び出し側が決める (既定は `Lama` と同じタイル毎の線形正規化。
    本番経路では膝つきトーンカーブを使う — docs/DESIGN_V2_JA.md §4.0)。
    """

    NORM_PCT = 99.9        # Lama と同じ。タイル毎の正規化基準
    # **自動露出の範囲**。クロップの [LO_PCT, HI_PCT] を [0,1] へ線形に写す。
    # 拡散モデルは学習分布 (普通の写真) から外れた暗く低コントラストな入力で崩れる。
    LO_PCT, HI_PCT = 0.5, 99.5

    def __init__(self, base: str = SD15_INPAINT, ckpt: str = YOEO_CKPT,
                 device: str = "mps", dtype=torch.float32, steps: int = 2,
                 vae_sample: bool = False, seed: int = 1234,
                 attn_slice: int | str | None = None,
                 auto_exposure: bool = True,
                 guidance: float = 1.0, negative: str = ""):
        from diffusers import AutoencoderKL, LCMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        base, ckpt = resolve(base), resolve(ckpt)
        for pth in (base, ckpt):
            if not os.path.exists(pth):
                raise FileNotFoundError(
                    f"{pth} が無い。取得方法は docs/DESIGN_V2_JA.md §7 Phase 0 を参照")

        self.device, self.dtype, self.steps = device, dtype, steps
        self.vae_sample, self.seed = vae_sample, seed
        self.auto_exposure = auto_exposure
        # **CFG で「描かせない」方向へ押す** (2026-08-25 の実験)。
        # 本家は guidance=1 (CFG 無し) だが、観測された失敗は
        # 「余計な人物を描く」こと。否定側に "person" を置けば
        # eps = eps_neg + g*(eps_cond - eps_neg) が人物から遠ざかる向きに働く。
        # g=1 なら従来どおり (否定側は使われない)。
        self.guidance, self.negative = guidance, negative

        self.vae = AutoencoderKL.from_pretrained(
            base, subfolder="vae", torch_dtype=dtype).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        self.te = CLIPTextModel.from_pretrained(
            base, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
        self.sched = LCMScheduler.from_pretrained(
            base, subfolder="scheduler", timestep_spacing="trailing")

        # UNet は config から作り、YOEO の EMA を**順序で**載せる
        self.unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(base, subfolder="unet")).to(dtype)
        ema = torch.load(ckpt, map_location="cpu", weights_only=False)
        shadow = ema["shadow_params"] if isinstance(ema, dict) else ema
        params = list(self.unet.parameters())
        if len(shadow) != len(params):
            raise RuntimeError(
                f"テンソル数が合わない: YOEO {len(shadow)} vs UNet {len(params)}")
        with torch.no_grad():
            for p, s in zip(params, shadow):
                if tuple(p.shape) != tuple(s.shape):
                    raise RuntimeError(f"形状不一致 {tuple(p.shape)} vs {tuple(s.shape)}")
                p.copy_(s.to(p.dtype))
        self.unet = self.unet.to(device).eval()

        # **1024² 以上では注意を分割しないと MPS のバッファ上限 (8GB) に当たる。**
        # 潜在 128² = 16,384 トークンの注意行列は 8 頭 fp32 で 8.6GB になり、
        # `RuntimeError: Invalid buffer size: 8.00 GiB` で落ちる。
        # `set_attention_slice` はクエリをチャンクに割って行列を実体化させない。
        if attn_slice is not None:
            self.unet.set_attention_slice(attn_slice)
        self.attn_slice = attn_slice

        # 空プロンプトの埋め込みを 1 度だけ作って使い回す (本家と同じ)
        with torch.inference_mode():
            ids = self.tok([""], padding="max_length",
                           max_length=self.tok.model_max_length,
                           truncation=True, return_tensors="pt").input_ids
            self.text_emb = self.te(ids.to(device))[0]
            nid = self.tok([negative], padding="max_length",
                           max_length=self.tok.model_max_length,
                           truncation=True, return_tensors="pt").input_ids
            self.neg_emb = self.te(nid.to(device))[0]

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        d = self.vae.encode(x).latent_dist
        z = d.sample(torch.Generator(device="cpu").manual_seed(self.seed)) \
            if self.vae_sample else d.mode()
        return z * self.vae.config.scaling_factor

    @torch.inference_mode()
    def fill(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`img` (1,3,H,W) [-1,1] と `mask` (1,1,H,W) [0,1] -> (1,3,H,W) [-1,1]。"""
        masked_latent = self._encode(img * (1 - mask))
        m_lat = F.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest-exact")

        self.sched.set_timesteps(timesteps=YOEO_TIMESTEPS[self.steps],
                                 device=self.device)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        lat = torch.randn(masked_latent.shape, generator=g).to(
            self.device, self.dtype) * self.sched.init_noise_sigma

        cfg = self.guidance > 1.0 + 1e-6
        for t in self.sched.timesteps:
            inp = torch.cat([lat, m_lat.to(lat), masked_latent.to(lat)], 1)
            if cfg:
                emb = torch.cat([self.neg_emb, self.text_emb], 0)
                e2 = self.unet(torch.cat([inp] * 2), t,
                               encoder_hidden_states=emb).sample
                e_neg, e_pos = e2.chunk(2)
                eps = e_neg + self.guidance * (e_pos - e_neg)
            else:
                eps = self.unet(inp, t, encoder_hidden_states=self.text_emb).sample
            lat = self.sched.step(eps, t, lat, generator=g).prev_sample

        out = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        return out.clamp(-1, 1)

    # ------------------------------------------------------------------
    def __call__(self, p: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """知覚空間で受けて知覚空間で返す (`Lama.__call__` と同じ契約)。

        **正規化は Lama と同じにしない。** Lama 流の `k = max(1.0, 99.9%tile)` は
        下限 1.0 のせいで「画像全体より暗いクロップ」を暗いまま渡してしまう。
        実測: 平均輝度 中央値 0.346 (普通の写真は 0.45〜0.55)、
        標準偏差 0.128 (普通は 0.20〜0.25)。60 件中 52% が暗すぎ、73% が低コントラスト。

        LaMa は畳み込みで意味的事前分布を持たないので耐えるが、
        **拡散モデルは学習分布から外れた入力で崩れる**。
        クロップの [0.5, 99.5] パーセンタイルを [0,1] へ線形に写し、厳密に逆写像する。
        マスク外は呼び出し側が原画素で上書きするので、逆写像の誤差は残らない。
        """
        if self.auto_exposure:
            lo = float(np.percentile(p, self.LO_PCT))
            hi = float(np.percentile(p, self.HI_PCT))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-4:
                lo, hi = 0.0, max(1.0, float(np.percentile(p, self.NORM_PCT)))
        else:                                   # 旧経路 (Lama と同じ流儀)
            lo, hi = 0.0, max(1.0, float(np.percentile(p, self.NORM_PCT)))

        s = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
        x = torch.from_numpy(s.transpose(2, 0, 1)[None].astype(np.float32))
        x = (x * 2.0 - 1.0).to(self.device, self.dtype)     # [-1,1]
        m = torch.from_numpy(mask[None, None].astype(np.float32)).to(
            self.device, self.dtype)
        y = self.fill(x, m)
        y = (y[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 0.5
        return (y * (hi - lo) + lo).astype(np.float32)


class SmartEraser:
    """SmartEraser (CVPR'25) による物体除去。`Lama` / `Yoeo` と同じ呼び出し方。

    **本家 `Model_framework/app_remove.py` から写したレシピ**

      - **Masked-Region Guidance**: `masked_image = init_image` —
        **物体を残したまま**渡す。標準 diffusers の
        `init_image * (mask_condition < 0.5)` をコメントアウトしただけの 1 行差。
        論文が FID 21.23 -> 10.90 の主因としているのがこれ。
        **専用パイプラインの持ち込みは不要** — 標準パイプラインに
        `masked_image_latents` を渡せば同じになる。
      - プロンプト: `'Remove the instance of'` を **max_length=7** でトークン化
      - 視覚特徴: マスク領域を**白背景で切り出し** (`crop_white_instance`)、
        CLIP ViT-L/14 の image_embeds -> MLP (`clip_mlp_weight.pth`) -> **7 トークン目**に差し込む
      - 既定: **50 step / guidance 1.5** (app の推奨値。ベンチ用スクリプトは 1.0)

    **速度**: 50 step は M1 で約 55 秒/箇所。YOEO の 2 step (5.5 秒) の 10 倍。

    **入力の正規化は `Yoeo` と同じ自動露出を使う。** Lama 流の
    `k = max(1.0, 99.9%tile)` は暗いクロップを暗いまま渡してしまい、
    拡散モデルが学習分布から外れる (YOEO で実測: 盲検A/B 25.6% -> 56.2%)。
    """

    LO_PCT, HI_PCT = 0.5, 99.5
    PROMPT = "Remove the instance of"
    TOK_LEN = 7

    def __init__(self, base: str = "models/smarteraser",
                 clip_dir: str = "models/clip-vit-large-patch14",
                 device: str = "mps", dtype=torch.float32, steps: int = 50,
                 guidance: float = 1.5, seed: int = 1234,
                 auto_exposure: bool = True):
        from diffusers import StableDiffusionInpaintPipeline
        from transformers import (CLIPImageProcessor, CLIPTokenizer,
                                  CLIPVisionModelWithProjection)

        base, clip_dir = resolve(base), resolve(clip_dir)
        for pth in (base, clip_dir):
            if not os.path.exists(pth):
                raise FileNotFoundError(f"{pth} が無い")

        self.device, self.dtype = device, dtype
        self.steps, self.guidance, self.seed = steps, guidance, seed
        self.auto_exposure = auto_exposure

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base, torch_dtype=dtype, safety_checker=None,
            requires_safety_checker=False).to(device)
        self.pipe.set_progress_bar_config(disable=True)

        self.tok = CLIPTokenizer.from_pretrained(clip_dir)
        self.proc = CLIPImageProcessor.from_pretrained(clip_dir)
        self.vision = CLIPVisionModelWithProjection.from_pretrained(
            clip_dir, torch_dtype=dtype).to(device).eval()

        # 本家 `IM2TEXT` は **残差ブロック**: `x + fc_out(norm(x))`。
        # (Linear -> GELU -> Linear だと推測して書いたら間違いだった。
        #  推測せず `modules/clip_visual_token.py` を読んで確認すること)
        self.mlp = _IM2TEXT().to(device, dtype)
        w = torch.load(os.path.join(base, "clip_mlp_weight.pth"),
                       map_location="cpu", weights_only=False)
        self.mlp.load_state_dict(w)
        self.mlp.eval()

    # ------------------------------------------------------------------
    def _embeds(self, crop_pil):
        """本家 `CLIPVisualPrompt.inference_vtoken` と同じ埋め込み構成。"""
        te, tk = self.pipe.text_encoder, self.tok
        ids = tk([self.PROMPT], padding="max_length", max_length=self.TOK_LEN,
                 truncation=True, return_tensors="pt").input_ids.to(self.device)
        uids = tk([""], padding="max_length", max_length=self.TOK_LEN,
                  truncation=True, return_tensors="pt").input_ids.to(self.device)
        th = te(ids).last_hidden_state
        uh = te(uids).last_hidden_state

        px = self.proc(images=crop_pil, return_tensors="pt")["pixel_values"]
        px = px.to(self.device, self.dtype)
        f = self.mlp(self.vision(px).image_embeds.unsqueeze(1))
        uf = self.mlp(self.vision(torch.zeros_like(px)).image_embeds.unsqueeze(1))

        pos = torch.cat((th[:, :-2, :], f, th[:, -1, :].unsqueeze(1)), 1)
        neg = torch.cat((uh[:, 0, :].unsqueeze(1), uf, uh[:, 1:-1, :]), 1)
        return pos, neg

    # ------------------------------------------------------------------
    def __call__(self, p: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageOps

        if self.auto_exposure:
            lo = float(np.percentile(p, self.LO_PCT))
            hi = float(np.percentile(p, self.HI_PCT))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-4:
                lo, hi = 0.0, max(1.0, float(np.percentile(p, 99.9)))
        else:
            lo, hi = 0.0, max(1.0, float(np.percentile(p, 99.9)))
        s = np.clip((p - lo) / (hi - lo), 0.0, 1.0)

        img_pil = Image.fromarray((s * 255 + 0.5).astype(np.uint8))
        m_pil = Image.fromarray((mask.astype(np.uint8) * 255))

        # crop_white_instance: マスク外を白にして対象の外接矩形で切り出す
        box = m_pil.getbbox()
        if box is None:
            return p.astype(np.float32)
        inv = ImageOps.invert(m_pil)
        white = inv.point(lambda x: 255 if x == 255 else 0).convert("RGB")
        crop = Image.composite(white, img_pil, inv).crop(box)

        with torch.inference_mode():
            pos, neg = self._embeds(crop)
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            out = self.pipe(prompt_embeds=pos, negative_prompt_embeds=neg,
                            image=img_pil, mask_image=m_pil,
                            num_inference_steps=self.steps,
                            guidance_scale=self.guidance,
                            generator=g, output_type="np").images[0]
        return (out.astype(np.float32) * (hi - lo) + lo).astype(np.float32)


class RORem:
    """RORem (CVPR'25) による物体除去。`Lama` / `Yoeo` と同じ呼び出し方。

    **なぜ試すのか (2026-08-30)**

    「自然な補完」で 6 本連続の空振り (step数 / マスク膨張 / Real-ESRGAN /
    人混みルーティング / CFG負プロンプト / 視野4倍) が出た。**全部が同じコア
    (YOEO) の周りのつまみ**だった。RORem は**コアそのものの入れ替え**なので軸が違う。

    論文の人手評価は成功率 **76.2% 対 LaMa 55.4%** で、候補中いちばん強い。
    ライセンスも Apache-2.0 で最も安全 (土台の SDXL は OpenRAIL++-M)。

    **重み**: `LetsThink/RORem` (HF)。**Google Drive 版ではなく完全な diffusers
    パイプライン**で、`unet/config.json` の `_name_or_path` が `RORem_mixed` を指す
    (= 512 と 1024 の両対応版)。`in_channels=9` を確認済み。
    土台の `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` は**別途不要**。

    **推論レシピ (本家 `inference_RORem.py` から書き写した)**

      - 50 step / `guidance_scale=1.0` / `strength=0.99`
      - プロンプトは固定文 (下記 `PROMPT`)。
        **`guidance_scale=1.0` では diffusers は CFG を回さない**ので、
        本家が併記している negative_prompt は実際には効いていない。
        よってここでも渡さない (渡しても無視されるだけ)。
      - マスクの膨張は本家既定 20px。ただし**当リポジトリのベンチのマスクは
        生成時に既に膨張済み**なので既定 0 とし、比較では両者に同一のマスクを渡す。

    **入力の正規化は `Yoeo` / `SmartEraser` と同じ自動露出**。
    暗いクロップをそのまま渡すと拡散モデルが学習分布から外れる
    (YOEO で実測: 盲検A/B 25.6% -> 56.2%)。

    **メモリ**: SDXL は fp16 でも重み 6.9GB。プロンプトは固定なので
    **埋め込みを 1 度だけ作ってテキストエンコーダを捨てる** (1.64GB 節約)。

    ---

    **採用しなかった (2026-08-30、盲検A/B)。再現用に残してある。**

    実除去ベンチで **YOEO 2step の 12勝5敗** (n=17、p=0.1435)。
    有意ではないので「RORem が劣る」は言えないが、
    **62.8 秒 対 5.3 秒 (12倍) を払って勝てていない**。
    論文の人手評価 (RORem 76.2% 対 LaMa 55.4%) は再現しなかった。

    負け方は **消した対象と同じ形の暗い残像**。継ぎ目は YOEO より良い
    (1.042 対 1.069) のに中身に残像が乗るので、自動指標では拾えない。

    2 つの機構仮説を立てて**両方とも実測で否定した** (eval/RESULTS.md に詳細):

      1. マスクが本家既定 (`dilate_size=20`) より細い
         -> 膨張しても残像は薄くなるだけで消えず、代わりに捏造が増える
      2. `strength=0.99` が初期潜在に元画像を漏らしている
         -> `strength=1.0` にしても残像は変わらない
         (出力自体は変わっていることを数値で確認済み: 穴の中の平均差 0.030)

    **残像はレシピの逸脱ではなくモデルの性質**という結論。
    """

    LO_PCT, HI_PCT = 0.5, 99.5
    PROMPT = ("4K, high quality, masterpiece, Highly detailed, Sharp focus, "
              "Professional, photorealistic, realistic")

    def __init__(self, base: str = "models/rorem", device: str = "mps",
                 dtype=torch.float16, steps: int = 50, guidance: float = 1.0,
                 strength: float = 0.99, dilate: int = 0, seed: int = 1234,
                 auto_exposure: bool = True, prompt: str | None = None):
        from diffusers import StableDiffusionXLInpaintPipeline

        base = resolve(base)
        if not os.path.exists(base):
            raise FileNotFoundError(
                f"{base} が無い。以下で取得 (Apache-2.0, 約 6.9GB):\n"
                "  python -c \"from huggingface_hub import snapshot_download;"
                "snapshot_download('LetsThink/RORem', local_dir='models/rorem')\"")

        self.device, self.dtype = device, dtype
        self.steps, self.guidance, self.strength = steps, guidance, strength
        self.dilate, self.seed = dilate, seed
        self.auto_exposure = auto_exposure

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base, torch_dtype=dtype).to(device)
        self.pipe.set_progress_bar_config(disable=True)

        # **プロンプトは固定なので 1 度だけ埋め込む。** SDXL のテキストエンコーダ 2 本は
        # fp16 でも 1.64GB あり、置いておくだけ無駄 (M1 16GB では効く)。
        txt = self.PROMPT if prompt is None else prompt
        with torch.inference_mode():
            pe, _, pp, _ = self.pipe.encode_prompt(
                prompt=txt, device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False)
        self._pe, self._pp = pe.clone(), pp.clone()
        self.pipe.text_encoder = None
        self.pipe.text_encoder_2 = None
        if device == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    def __call__(self, p: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image

        if self.auto_exposure:
            lo = float(np.percentile(p, self.LO_PCT))
            hi = float(np.percentile(p, self.HI_PCT))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-4:
                lo, hi = 0.0, max(1.0, float(np.percentile(p, 99.9)))
        else:
            lo, hi = 0.0, max(1.0, float(np.percentile(p, 99.9)))
        s = np.clip((p - lo) / (hi - lo), 0.0, 1.0)

        m = mask.astype(bool)
        if self.dilate:
            # 本家 `myutils/img_util.dilate_mask` と同じ矩形カーネルの膨張
            import scipy.ndimage as ndi
            m = ndi.binary_dilation(m, np.ones((3, 3), bool), self.dilate)
        if m.sum() == 0:
            return p.astype(np.float32)

        H, W = s.shape[:2]
        img_pil = Image.fromarray((s * 255 + 0.5).astype(np.uint8))
        m_pil = Image.fromarray(m.astype(np.uint8) * 255)

        with torch.inference_mode():
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            out = self.pipe(
                prompt_embeds=self._pe, pooled_prompt_embeds=self._pp,
                image=img_pil, mask_image=m_pil, height=H, width=W,
                num_inference_steps=self.steps, guidance_scale=self.guidance,
                strength=self.strength, generator=g,
                output_type="np").images[0]
        if not np.isfinite(out).all():
            raise RuntimeError(
                "RORem の出力に NaN。SDXL の VAE を fp16 で回すと出ることがある。"
                " dtype=torch.float32 で作り直すこと")
        return (out.astype(np.float32) * (hi - lo) + lo).astype(np.float32)


class _IM2TEXT(torch.nn.Module):
    """本家 `Model_framework/modules/clip_visual_token.py` の IM2TEXT をそのまま写す。

        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.norm   = nn.LayerNorm(output_dim)
        forward: return x + self.fc_out(self.norm(x))
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.fc_out = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc_out(self.norm(x))


def composite(raw: np.ndarray, img: np.ndarray, mask: np.ndarray,
              poisson: bool = True) -> np.ndarray:
    """L5: 除去器の生出力を写真に戻す (docs/DESIGN_V2_JA.md の L5)。

    **VAE を通す除去器は、そのまま貼ると必ず継ぎ目が見える。**
    実測 (実除去ベンチ 60 件、`boundary_discontinuity`、1.0 が「見えない」):

        LaMa  中央値 1.31 / 1.5 超え 18/60
        YOEO  中央値 1.89 / 1.5 超え 55/60   <- ほぼ全ケースで継ぎ目

    LaMa の生出力はマスク外で原画像とほぼ一致する (差 2.47e-05) ので
    合成が原理的に継ぎ目なしになるが、YOEO はマスク内外が別物になるため。
    ユーザが「周りと馴染んでない色がマスクと同じ形になっている」と指摘したのはこれ。

    `poisson_blend` は勾配を合わせるので継ぎ目が原理的に消える。実測:

        素 1.90 -> poisson 1.06 (24/24 件で改善、LaMa の 1.31 より良い)

    羽ぼかしは 0.92 まで下がるが、これは**逆に滑らかすぎる** (周囲より平坦)。
    Poisson の 1.06 が理想の 1.00 に最も近いので既定とする。

    **LaMa に掛けても実質無効** (生出力がマスク外で原画像と一致するため
    食い違いが 0 になり調和膜が消える)。両者に同じ処理を掛けてよい = 公平。
    """
    from nagi_inpaint2.stage_d import poisson_blend
    fill = poisson_blend(raw, img, mask) if poisson else raw
    return np.where(mask[..., None], fill, img)


def build(name: str, device: str = "mps", **kw):
    """名前から除去器を作る。`eval/*` から使う入口。"""
    if name == "lama":
        from nagi_inpaint2.lama import Lama
        return Lama(device=device)
    if name == "yoeo":
        return Yoeo(device=device, **kw)
    if name == "smarteraser":
        return SmartEraser(device=device, **kw)
    if name == "rorem":
        return RORem(device=device, **kw)
    raise ValueError(
        f"未知の除去器: {name!r} (lama / yoeo / smarteraser / rorem)")
