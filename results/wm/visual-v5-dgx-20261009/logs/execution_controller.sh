#!/usr/bin/env bash
set -euo pipefail
work=/aifs4su/guhao/KVCache/WM
cd "$work/code"
exec 9>"$work/controller.lock"
flock -n 9
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$work/cache/matplotlib"
py="$work/.venv/bin/python"
state() {
  python3 - "$work" "$1" "${2:-}" <<'PY'
import json,sys,time,os
from pathlib import Path
p=Path(sys.argv[1])/'controller_status.json'
v=dict(stage=sys.argv[2],detail=sys.argv[3],updated_unix=time.time(),pid=os.getppid())
t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
PY
}
failed() { rc=$?; state failed "controller exit=$rc; inspect logs and preserve all task files"; exit "$rc"; }
trap failed ERR
state waiting_preparation
for ((i=0; i<240; i++)); do
  [[ -f "$work/preparation.json" ]] && break
  if ! kill -0 "$(cat "$work/logs/preparation.pid")" 2>/dev/null; then
    echo 'Preparation process stopped without completion; retaining workspace' >&2
    state failed 'preparation process stopped without completion; retain workspace'
    exit 1
  fi
  sleep 15
done
[[ -f "$work/preparation.json" ]]
[[ "$(git rev-parse HEAD)" = d0b4fef25e98eefb58bfba737997cb0145f27daa ]]
git diff --exit-code
state verifying_preparation
"$py" - "$work" <<'PY'
import json,sys
from pathlib import Path
from pmr.visual_memory.common import digest
root=Path(sys.argv[1]); prep=json.loads((root/'preparation.json').read_text())
assert prep['status']=='complete'
assert digest(root/'assets/IN1K-vit.h.14-300e.pth.tar')==prep['checkpoint_sha256']
PY
args=(--backend dgx --account hansirui --partition llm --qos llm --gpu-model H800 --allocation-job 415453 --shared-allocation-gpus 8 --gpu-index 3 --parallel-gpus 8 --allow-gpu-sharing --out "$work/run1" --assets "$work/assets")
state dry_run
"$py" scripts/launch_visual_v5.py "${args[@]}" --all > "$work/logs/dry-run.log" 2>&1
state profile
"$py" scripts/launch_visual_v5.py "${args[@]}" --submit > "$work/logs/profile-controller.log" 2>&1
"$py" - "$work/run1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); profile=json.loads((p/'profile.json').read_text());status=json.loads((p/'profile_status.json').read_text())
assert profile['status']==status['status']=='complete' and profile['throughput_gate_passed']
assert 'H800' in profile['runtime']['gpu']
PY
state full_chain
if "$py" scripts/launch_visual_v5.py "${args[@]}" --all --resume --submit > "$work/logs/chain-controller.log" 2>&1; then
  chain_rc=0
else
  chain_rc=$?
fi
state diagnostics "chain exit=$chain_rc"
if [[ -f "$work/run1/feature_gate.json" ]]; then
  "$py" scripts/diagnose_visual_features.py --out "$work/run1" --threads 1 > "$work/logs/feature-diagnostics.log" 2>&1
fi
"$py" -m pmr.visual_memory report --out "$work/run1" > "$work/logs/report.log" 2>&1
sacct -j 415453 -nP --format=JobID,State,ElapsedRaw,AllocTRES > "$work/logs/slurm-accounting-final.txt"
if [[ "$chain_rc" -ne 0 ]]; then
  state failed "scientific chain exit=$chain_rc; diagnostics retained; no cleanup"
  exit "$chain_rc"
fi
"$py" - "$work/run1" <<'PY'
import json,sys
from pathlib import Path
assert json.loads((Path(sys.argv[1])/'report_status.json').read_text())['status']=='pilot_complete'
PY
state experiment_complete "Await local GitHub archival and download verification; no remote cleanup here"
