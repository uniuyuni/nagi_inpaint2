"""重みを揃える CLI。ShadeWave の setup.sh から呼ばれる。

  python -m nagi_inpaint2.fetch --dest checkpoints/nagi_inpaint2
  python -m nagi_inpaint2.fetch --dest checkpoints/nagi_inpaint2 --check        # 確認だけ
  python -m nagi_inpaint2.fetch --dest DIR --mirror ~/LibraryProjects/nagi_inpaint2/models

終了コード: 0 = 揃った / 1 = 揃っていない (取得失敗を含む)
"""
from __future__ import annotations

import argparse
import sys

from nagi_inpaint2.assets import ASSETS, TOTAL_BYTES, check, fetch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nagi_inpaint2.fetch")
    ap.add_argument("--dest", required=True, help="重みを置くディレクトリ")
    ap.add_argument("--mirror", help="同じ構成のディレクトリがあればそこから流用する")
    ap.add_argument("--check", action="store_true", help="取得せず、揃っているかだけ見る")
    ap.add_argument("--deep", action="store_true", help="--check で sha256 まで照合する")
    a = ap.parse_args(argv)

    if a.check:
        bad = check(a.dest, deep=a.deep)
        for b in bad:
            print(f"不足/不一致: {b.path}")
        print(f"{len(ASSETS) - len(bad)}/{len(ASSETS)} 件そろっている")
        return 0 if not bad else 1

    print(f"nagi_inpaint2 の重み {len(ASSETS)} 件 / 合計 {TOTAL_BYTES/1e9:.2f}GB -> {a.dest}")
    failed = fetch(a.dest, mirror=a.mirror)
    if failed:
        print(f"エラー: {len(failed)} 件を取得できなかった", file=sys.stderr)
        return 1
    print("nagi_inpaint2 の重みを配置した")
    return 0


if __name__ == "__main__":
    sys.exit(main())
