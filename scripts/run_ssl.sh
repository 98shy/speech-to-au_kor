#!/bin/bash
cd /c/Users/user/Documents/IITP-29/experiment/korean_train
PY=/c/Users/user/miniconda3/envs/iitp/python.exe
$PY -I extract_feats_ssl.py utter-project/mHuBERT-147 9 mhubert > log_extract_mhubert.txt 2>&1 || echo "EXTRACT mhubert FAILED"
$PY -I extract_feats_ssl.py facebook/wav2vec2-xls-r-300m 15 xlsr > log_extract_xlsr.txt 2>&1 || echo "EXTRACT xlsr FAILED"
for feat in mhubert xlsr; do
  for f in 0 1 2 3 4; do
    tag=joint_tcn_scratch_fold${f}_feat${feat}
    [ -f results_${tag}.csv ] || FEAT=$feat $PY -I train_korean.py joint tcn scratch 30 $f > log_${tag}.txt 2>&1
    echo "finished $tag"
  done
done
echo SSL ALL DONE
