#!/bin/bash
cd /c/Users/user/Documents/IITP-29/experiment/korean_train
PY=/c/Users/user/miniconda3/envs/iitp/python.exe
go() { # tag  then env assignments passed via env, args after
  local tag="$1"; shift
  [ -f results_${tag}.csv ] || "$@" > log_${tag}.txt 2>&1
  echo "finished $tag"
}
# Exp1: Olkavs label shift +70 ms (K-MIND lag ~0 -> unchanged)
go olkavs_tcn_scratch_lag70 env LAG_OLKAVS=0.07 TAGX=_lag70 $PY -I train_korean.py olkavs tcn scratch 30
for f in 0 1 2 3 4; do go joint_tcn_scratch_fold${f}_lag70 env LAG_OLKAVS=0.07 TAGX=_lag70 $PY -I train_korean.py joint tcn scratch 30 $f; done
# Exp2: training-label smoothing sigma=2 frames (80 ms); test labels stay raw
go olkavs_tcn_scratch_smooth2 env SMOOTH=2 TAGX=_smooth2 $PY -I train_korean.py olkavs tcn scratch 30
for f in 0 1 2 3 4; do go kmind_tcn_scratch_fold${f}_smooth2 env SMOOTH=2 TAGX=_smooth2 $PY -I train_korean.py kmind tcn scratch 30 $f; done
# Exp3: SET003-5 rotation (test/val clean per-speaker audio); train = 1 clean set, with vs without SET001/002 (MIX audio)
for r in 0 1 2; do
  go kmind_tcn_scratch_fold${r}_cleanonly env CLEAN_ROT=1 TAGX=_cleanonly $PY -I train_korean.py kmind tcn scratch 30 $r
  go kmind_tcn_scratch_fold${r}_cleanplusmix env CLEAN_ROT=1 INCL_MIX=1 TAGX=_cleanplusmix $PY -I train_korean.py kmind tcn scratch 30 $r
done
echo EXP123 ALL DONE
