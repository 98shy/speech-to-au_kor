#!/bin/bash
cd /c/Users/user/Documents/IITP-29/experiment/korean_train
PY=/c/Users/user/miniconda3/envs/iitp/python.exe
for f in 0 1 2 3 4; do
  for m in kmind joint; do
    [ -f results_${m}_tcn_scratch_fold${f}.csv ] || $PY -I train_korean.py $m tcn scratch 30 $f > log_${m}_tcn_scratch_fold${f}.txt 2>&1
    echo "finished $m fold$f"
  done
done
echo CV ALL DONE
