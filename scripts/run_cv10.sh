#!/bin/bash
cd /c/Users/user/Documents/IITP-29/experiment/korean_train
PY=/c/Users/user/miniconda3/envs/iitp/python.exe
for f in 0 1 2 3 4 5 6 7 8 9; do
  tag=joint_tcn_scratch_fold${f}_all10
  [ -f results_${tag}.csv ] || env TAGX=_all10 NSETS=10 $PY -I train_korean.py joint tcn scratch 30 $f > log_${tag}.txt 2>&1
  echo "finished $tag"
done
for f in 0 1 2 3 4 5 6 7 8 9; do
  tag=joint_tcn_scratch_fold${f}_all10_au12f
  [ -f results_${tag}.csv ] || env TAGX=_all10_au12f NSETS=10 AU12_MINPREV=0.05 $PY -I train_korean.py joint tcn scratch 30 $f > log_${tag}.txt 2>&1
  echo "finished $tag"
done
echo CV10 ALL DONE
