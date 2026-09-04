"""Day 3 / 索引構築 — 全 RAF の EXIF と隣接 JPEG を索引化する (SPEC §6.4 step 1)。

RAW の本現像は行わない。exiftool のヘッダ読みのみ。

出力: cache/index.jsonl (1行1ファイル)。中断・再開可能。

  pixi run python data/index.py
  pixi run python data/index.py --limit 500        # 動作確認
  pixi run python data/index.py --bodies X-T5      # 機種を絞る

注意: /Volumes/PHOTOS 01 は読み出しのみ。一切書き込まない。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = "/Volumes/PHOTOS 01/Photos"
BODIES = ["X-E3", "X-T4", "X-T5"]
OUT = "cache/index.jsonl"
CHUNK = 2000

# exiftool から取り出すタグ。順序が下の KEYS と対応する。
# NOTE: -SourceFile は -T (tab) モードでは常に "-" を返す疑似タグなので使えない。
#       -Directory と -FileName を組み合わせてパスを復元する。
TAGS = [
    "Directory", "FileName", "Model", "ISO", "FNumber", "ExposureTime",
    "FocalLength", "RawImageFullSize", "DateTimeOriginal", "LensModel",
    "Orientation", "ShutterCount",
]
KEYS = [
    "dir", "filename", "model", "iso", "fnumber", "exposure",
    "focal", "raw_size", "datetime", "lens",
    "orientation", "shutter_count",
]


def p(*a):
    print(*a, flush=True)


def find_rafs(root: str, bodies: list[str]) -> list[str]:
    out = []
    for b in bodies:
        d = os.path.join(root, b)
        if not os.path.isdir(d):
            p(f"  警告: {d} が存在しない")
            continue
        for dp, _, fns in os.walk(d):
            for f in fns:
                if f.lower().endswith(".raf"):
                    out.append(os.path.join(dp, f))
    return sorted(out)


EXTS = (".JPG", ".jpg", ".JPEG", ".jpeg")


def sibling_jpeg(raf: str) -> str | None:
    """同名の JPEG。スコアリング (data/score.py) で RAF の代わりに読む。

    32,787枚の実測内訳:
      同ディレクトリ 32,348 / 親ディレクトリ 71 / 見つからない 368 (1.1%)

    親ディレクトリを見るのは、一部のショットで RAF だけが `<shoot>/RAW/`
    サブディレクトリに入っており、JPEG が `<shoot>/` 直下にあるため
    (72件。うち 3件はディレクトリ名が " RAW" と先頭に空白が入っている)。
    見つからない場合は None。score.py が RAF の埋め込みプレビューに退避する。
    """
    stem = os.path.splitext(os.path.basename(raf))[0]
    d = os.path.dirname(raf)
    for base in (os.path.join(d, stem), os.path.join(os.path.dirname(d), stem)):
        for ext in EXTS:
            if os.path.exists(base + ext):
                return base + ext
    return None


def shoot_of(raf: str, root: str) -> tuple[str, str]:
    """(機種, ショットフォルダ名) を返す。

    構成は <root>/<機種>/<年>/<YYYY-MM-DD 地名>/xxx.RAF。
    ショットフォルダ名は §6.4 の層別サンプリングと §9.2 の擬似ペア発掘に使う
    (同一フォルダ = 同一ショット)。
    """
    rel = os.path.relpath(raf, root)
    parts = rel.split(os.sep)
    body = parts[0] if parts else ""
    shoot = parts[2] if len(parts) >= 4 else (parts[1] if len(parts) >= 3 else "")
    return body, shoot


def run_exiftool(paths: list[str]) -> list[dict]:
    args = ["exiftool", "-fast2", "-q", "-n", "-T"]
    args += [f"-{t}" for t in TAGS]
    proc = subprocess.run(args + paths, capture_output=True, text=True)
    rows = []
    for line in proc.stdout.splitlines():
        vals = line.split("\t")
        if len(vals) != len(KEYS):
            continue
        rows.append({k: (None if v == "-" else v) for k, v in zip(KEYS, vals)})
    return rows


def enrich(rec: dict, root: str) -> dict:
    raf = os.path.join(rec.pop("dir") or "", rec.pop("filename") or "")
    rec["path"] = raf
    body, shoot = shoot_of(raf, root)
    jpg = sibling_jpeg(raf)
    try:
        size = os.path.getsize(raf)
    except OSError:
        size = None
    for k in ("iso", "orientation", "shutter_count"):
        try:
            rec[k] = int(float(rec[k])) if rec[k] is not None else None
        except (TypeError, ValueError):
            rec[k] = None
    for k in ("fnumber", "exposure", "focal"):
        try:
            rec[k] = float(rec[k]) if rec[k] is not None else None
        except (TypeError, ValueError):
            rec[k] = None
    rec.update(body=body, shoot=shoot, jpeg=jpg, raf_bytes=size)
    return rec


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
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--bodies", nargs="*", default=BODIES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    p(f"走査中: {a.root} {a.bodies}")
    rafs = find_rafs(a.root, a.bodies)
    if a.limit:
        rafs = rafs[:a.limit]
    p(f"  RAF {len(rafs):,} 件")

    done = load_done(a.out)
    todo = [r for r in rafs if r not in done]
    if done:
        p(f"  処理済み {len(done):,} 件をスキップ -> 残り {len(todo):,} 件")
    if not todo:
        p("完了済み")
        return 0

    t0 = time.perf_counter()
    n_jpeg = 0
    with open(a.out, "a") as fo:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            for rec in run_exiftool(chunk):
                rec = enrich(rec, a.root)
                n_jpeg += rec["jpeg"] is not None
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            n = min(i + CHUNK, len(todo))
            el = time.perf_counter() - t0
            p(f"  {n:>6,}/{len(todo):,}  {el/60:5.1f}分経過  "
              f"残り約 {el/n*(len(todo)-n)/60:5.1f}分")

    p(f"\n完了: {a.out}")
    p(f"  隣接 JPEG あり: {n_jpeg:,} / {len(todo):,} "
      f"({n_jpeg/max(len(todo),1)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
