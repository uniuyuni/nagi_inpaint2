"""推論に必要な重みの一覧と、その取得・検証 (**torch を import しない**)。

**なぜここに置くのか**: ShadeWave の setup.sh は nagi_denoise と同じく、
重みのファイル名や URL を**シェルスクリプトに直書きしない**。一覧はこのモジュールが持ち、
setup.sh は `python -m nagi_inpaint2.fetch` を呼ぶだけにする。
一覧と実装がずれても、ここだけ直せば済む。

**一覧の決め方**: 実測した構成 (採用率 88.3%) で**実際に読まれているファイルだけ**。

  - SD1.5-inpainting は VAE / テキストエンコーダ / tokenizer / scheduler の設定だけ使う。
    **UNet の重みは要らない** (YOEO が置き換える。`unet/config.json` だけで構成する)
  - 上流には同じ重みが 3 形式ある (.bin / fp16.bin / fp16.safetensors)。
    実測した構成は **fp32 の `.bin`** を読んでいたので、それだけを取る。
    fp16 版に替えると出力が変わり、採用率の実測の前提が崩れる
  - 上流が後から差し替えても結果が変わらないよう、**リビジョンを固定し sha256 で検証**する

合計 4.47GB。
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    path: str                    # 置き場所 (重みディレクトリからの相対)
    size: int
    sha256: str
    url: str | None = None       # 直接ダウンロード
    hf: tuple[str, str, str] | None = None   # (repo_id, revision, リポジトリ内のパス)

    @property
    def license(self) -> str:
        if self.path.startswith("sd15-inpaint/"):
            return "CreativeML OpenRAIL-M"
        return "Apache-2.0"


ASSETS: tuple[Asset, ...] = (
    Asset("big-lama.pt", 205803670, "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c",
          url="https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"),
    Asset("yoeo/yoeo.pth", 3438338334, "a743efd708907b1f0bb9fbb375e80ab6548f4f0d5a74fcd25e9cb64e6146be3b",
          hf=("unhzyx/yoeo", "dccaa88a0154ed5f8040bc7c26b04a5b07b27fcc", "yoeo.pth")),
    Asset("sd15-inpaint/model_index.json", 548, "636d700de8970a70f02b91d433ad949dc2ceb750f39ba0cf2cf39ef0b0b9d1e4",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "model_index.json")),
    Asset("sd15-inpaint/scheduler/scheduler_config.json", 313, "8b457848b74deeeab2a967b75048dd2cb6f393cfc17751eca1b5ec655cff0183",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "scheduler/scheduler_config.json")),
    Asset("sd15-inpaint/unet/config.json", 748, "669f6581087fac5b16c44e42c58b359a35eadf2af5691d7945480bd7ceac8e6d",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "unet/config.json")),
    Asset("sd15-inpaint/feature_extractor/preprocessor_config.json", 342, "2a1da83b5e1032aaeef397552ddb408dca0d8cd1dc58f61bf6abf38d6f33a0a2",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "feature_extractor/preprocessor_config.json")),
    Asset("sd15-inpaint/text_encoder/config.json", 617, "845df614cb9327ae7bbea027316246fae917827407da6df13572e41b5f93b4cc",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "text_encoder/config.json")),
    Asset("sd15-inpaint/text_encoder/pytorch_model.bin", 492305335, "770a47a9ffdcfda0b05506a7888ed714d06131d60267e6cf52765d61cf59fd67",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "text_encoder/pytorch_model.bin")),
    Asset("sd15-inpaint/tokenizer/merges.txt", 524619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "tokenizer/merges.txt")),
    Asset("sd15-inpaint/tokenizer/special_tokens_map.json", 472, "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "tokenizer/special_tokens_map.json")),
    Asset("sd15-inpaint/tokenizer/tokenizer_config.json", 806, "00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "tokenizer/tokenizer_config.json")),
    Asset("sd15-inpaint/tokenizer/vocab.json", 1059962, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "tokenizer/vocab.json")),
    Asset("sd15-inpaint/vae/config.json", 552, "4cd0ce4803567837175fbb8412689fe77d069c1ef8171beb83979db309a8bad5",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "vae/config.json")),
    Asset("sd15-inpaint/vae/diffusion_pytorch_model.bin", 334707217, "1b134cded8eb78b184aefb8805b6b572f36fa77b255c483665dda931fa0130c5",
          hf=("stable-diffusion-v1-5/stable-diffusion-inpainting", "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb", "vae/diffusion_pytorch_model.bin")),
)

TOTAL_BYTES = sum(a.size for a in ASSETS)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check(dest: str, deep: bool = False) -> list[Asset]:
    """揃っていないものを返す。`deep=True` なら sha256 まで照合する (3.4GB で約 10 秒)。

    既定は大きさだけを見る。起動のたびにハッシュを取るのは重すぎるため。
    """
    bad = []
    for a in ASSETS:
        f = os.path.join(dest, a.path)
        if not os.path.isfile(f) or os.path.getsize(f) != a.size:
            bad.append(a)
        elif deep and sha256_of(f) != a.sha256:
            bad.append(a)
    return bad


def _place(src: str, dst: str) -> None:
    """同じボリュームならハードリンク (容量を食わない)、だめならコピー。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _download(a: Asset, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if a.hf is not None:
        from huggingface_hub import hf_hub_download
        repo, rev, name = a.hf
        p = hf_hub_download(repo, name, revision=rev)
        _place(os.path.realpath(p), dst)
        return
    import urllib.request
    part = dst + ".part"
    with urllib.request.urlopen(a.url) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 24)
    os.replace(part, dst)


def fetch(dest: str, mirror: str | None = None, log=print) -> list[Asset]:
    """足りないものを取ってきて検証する。戻り値は失敗したもの (空なら成功)。

    `mirror` に同じ構成のディレクトリがあれば、ダウンロードの代わりにそこから置く
    (開発機で nagi_inpaint2 の `models/` を流用する場合など)。
    **置いたものは必ず sha256 で検証する。** 壊れたファイルを黙って使わないため。
    """
    failed = []
    todo = check(dest)
    if not todo:
        log(f"nagi_inpaint2 の重みは揃っている: {dest}")
        return []
    for a in todo:
        dst = os.path.join(dest, a.path)
        src = os.path.join(mirror, a.path) if mirror else None
        try:
            if src and os.path.isfile(src) and os.path.getsize(src) == a.size:
                log(f"  流用  {a.path}  ({a.size/1e6:.0f}MB)")
                _place(src, dst)
            else:
                log(f"  取得  {a.path}  ({a.size/1e6:.0f}MB)")
                _download(a, dst)
            got = sha256_of(dst)
            if got != a.sha256:
                os.remove(dst)
                raise ValueError(f"sha256 が一致しない: {got[:12]}… (期待 {a.sha256[:12]}…)")
        except Exception as e:      # 1 つの失敗で残りを止めない。最後にまとめて報告する
            log(f"  失敗  {a.path}: {type(e).__name__}: {e}")
            failed.append(a)
    return failed
