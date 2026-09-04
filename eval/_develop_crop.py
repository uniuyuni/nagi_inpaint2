"""RAW を現像し中央クロップを .npy に書く (別プロセス用)。

torch と libraw_enhanced はそれぞれ別の libomp をリンクしており、
同一プロセスに同居させると OMP Error #15 になる。
KMP_DUPLICATE_LIB_OK で握り潰すと結果が壊れ得るので、プロセスを分ける。

  python eval/_develop_crop.py <raw> <crop> <out.npy>
"""
import sys
import numpy as np
import libraw_enhanced as lre
sys.path.insert(0, ".")
from data.extract import POSTPROCESS_KW
from nagi_inpaint2.color import estimate_white, to_perceptual

raw_path, crop, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with lre.imread(raw_path) as raw:
    img = raw.postprocess(**POSTPROCESS_KW)
white = estimate_white(img)                      # white は画像全体で決める (§3.3)
p = to_perceptual(img, white).astype(np.float32)
h, w = p.shape[:2]
y0, x0 = (h - crop) // 2, (w - crop) // 2
np.save(out, np.ascontiguousarray(p[y0:y0 + crop, x0:x0 + crop]))
print(f"{p.shape} white={white:.5f} -> {out}")
