#!/bin/bash
# 進行状況の一覧表示。いつでも実行してよい (読み取りのみ、学習には影響しない)。
#   bash status.sh
cd "$(dirname "$0")"
PY=.pixi/envs/default/bin/python
BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'
G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'

echo "${BOLD}== 実行中のジョブ ==${OFF}  $(date '+%m/%d %H:%M')"
found=0
while IFS= read -r line; do
  pid=$(echo "$line" | awk '{print $2}')
  cpu=$(echo "$line" | awk '{print $3}')
  el=$(echo "$line" | awk '{print $10}')
  cmd=$(echo "$line" | sed 's/.*bin\/python //' | cut -c1-72)
  printf "  ${G}●${OFF} %-72s ${DIM}pid %s / cpu %s%% / 経過 %s${OFF}\n" "$cmd" "$pid" "$cpu" "$el"
  found=1
done < <(ps aux | grep -E "[t]rain_.*\.py|[e]val/benchmark\.py|[e]val/compare_visual\.py|[d]ata/.*\.py" )
[ $found -eq 0 ] && echo "  ${DIM}(なし)${OFF}"

echo
echo "${BOLD}== 学習の進捗 ==${OFF}"
for h in checkpoints/*/history.jsonl; do
  [ -f "$h" ] || continue
  # 直近5分以内に更新されたものだけを「進行中」として詳しく出す
  age=$(( $(date +%s) - $(stat -f %m "$h") ))
  [ $age -gt 1800 ] && continue
  name=$(basename "$(dirname "$h")")
  $PY - "$h" "$name" "$age" <<'PYEOF'
import json, sys
path, name, age = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(path) if l.strip()]
if not rows:
    sys.exit()
r = rows[-1]
step = r.get("step", 0)
sec = r.get("sec_per_step")
# 総 step は履歴からは分からないので、直近の伸びから推定表示に留める
bar_n = 28
tot = None
for guess in (8000, 12000, 30000, 40000):
    if step <= guess:
        tot = guess; break
tot = tot or step
fill = int(bar_n * step / tot)
bar = "█" * fill + "░" * (bar_n - fill)
eta = ""
if sec and step < tot:
    m = int((tot - step) * sec / 60)
    eta = f" / 残り約 {m//60}時間{m%60}分" if m >= 60 else f" / 残り約 {m}分"
print(f"  \033[1m{name}\033[0m  {bar} {step}/{tot}{eta}")
keys = [("l1","l1"),("perc","知覚"),("adv","敵対"),("d_loss","識別器"),("dw","|Δw|max")]
line = "    " + "  ".join(f"{lab} {r[k]:.4f}" for k, lab in keys if k in r)
print(line)
vals = [(x["step"], x.get("val_perc") or x.get("val_l1")) for x in rows if x.get("val_perc") or x.get("val_l1")]
if vals:
    tail = vals[-5:]
    print("    保留: " + " → ".join(f"{v:.5f}" for _, v in tail))
print(f"    \033[2m最終更新 {age}秒前\033[0m")
PYEOF
done

echo
echo "${BOLD}== 直近の測定結果 ==${OFF} ${DIM}(eval/RESULTS.md の末尾)${OFF}"
grep -E "^\|" eval/RESULTS.md 2>/dev/null | tail -6 | sed 's/^/  /'

echo
echo "${BOLD}== 生成済みの比較画像 ==${OFF}"
ls -t out/compare/*.png 2>/dev/null | head -5 | while read f; do
  printf "  ${C}%s${OFF}  ${DIM}%s${OFF}\n" "$f" "$(stat -f '%Sm' -t '%m/%d %H:%M' "$f")"
done
[ -z "$(ls out/compare/*.png 2>/dev/null)" ] && echo "  ${DIM}(なし)${OFF}"
echo
