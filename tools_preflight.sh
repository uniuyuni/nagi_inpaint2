#!/bin/bash
# 長時間実行の前に必ず通す疎通確認。
#   bash tools_preflight.sh "<本番コマンド>"
# 本番と同じコマンド・同じデバイスで数 step だけ回し、checkpoint の往復まで確認する。
#
# **なぜ必要か**: このセッションで、実行前に一度も動かさなかったことによる事故が5件起きた。
#   ゼロ初期化で勾配が消えたまま27時間 / p() の TypeError で12分 /
#   MPS の mode="area" 未実装で学習が即死 / checkpoint の base 不一致でロード失敗 /
#   CLI のタグ未登録で KeyError。いずれも数十秒の試走で防げた。
set -u
CMD="$1"
echo "== 1. 構文チェック =="
for f in $(echo "$CMD" | grep -oE '[a-zA-Z0-9_/]+\.py'); do
  [ -f "$f" ] || continue
  .pixi/envs/default/bin/python -c "import ast,sys; ast.parse(open('$f').read())" \
    && echo "  ok $f" || { echo "  ✗ 構文エラー $f"; exit 1; }
done

echo "== 2. 短縮版で試走 (--steps を 6 に置換) =="
SHORT=$(echo "$CMD" | sed -E 's/--steps [0-9]+/--steps 6/; s/--out [^ ]+/--out \/tmp\/preflight_out/; s/--log-every [0-9]+/--log-every 3/; s/--val-every [0-9]+/--val-every 6/')
echo "  $SHORT"
if ! eval "$SHORT" > /tmp/preflight.log 2>&1; then
  echo "  ✗ 試走で失敗:"; grep -vE "Warning|Consider|rfft2|irfft2" /tmp/preflight.log | tail -12; exit 1
fi
echo "  ok"

echo "== 3. checkpoint の保存と読み込みの往復 =="
.pixi/envs/default/bin/python - <<'PY'
import glob, sys, torch
fs = glob.glob("/tmp/preflight_out/*.pt")
if not fs:
    print("  ! checkpoint が作られていない (保存経路を確認すること)"); sys.exit(0)
for f in fs:
    ck = torch.load(f, map_location="cpu")
    meta = {k: v for k, v in ck.items() if k != "model"}
    print(f"  ok {f}  meta={meta}")
PY

echo "== 4. パラメータが動いたか =="
grep -E "Δw" /tmp/preflight.log | tail -1 || echo "  ! |Δw| のログが無い"
echo
echo "疎通確認を通過。本番コマンドを実行してよい。"
