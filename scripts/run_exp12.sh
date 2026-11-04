#!/bin/bash
cd /c/Users/user/Documents/IITP-29/experiment/korean_train
PY=/c/Users/user/miniconda3/envs/iitp/python.exe
run() { # name mode init fold [olkN]
  local tag="$1_tcn_$2$([ -n "$3" ] && echo _fold$3)$([ -n "$4" ] && echo _olkN$4)"
  [ -f results_${tag}.csv ] || OLK_N=$4 $PY -I train_korean.py $1 tcn $2 30 $3 > log_${tag}.txt 2>&1
  echo "finished $tag"
}
# experiment 2: English-checkpoint fine-tuning vs scratch
for f in 0 1 2 3 4; do run kmind finetune $f ""; done
run olkavs finetune "" ""
# experiment 1: Olkavs data-amount curve (olkavs-only and joint), N=10 already done as scratch
for n in 2 4 6 8; do run olkavs scratch "" $n; run joint scratch "" $n; done
echo EXP12 ALL DONE
