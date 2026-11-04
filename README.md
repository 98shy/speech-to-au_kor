# speech-to-au_kor (temporary snapshot)

Korean-speech training/evaluation code for speech -> facial AU intensity (AU12, AU25, AU26, 25 fps).
Companion to [speech-to-au](https://github.com/98shy/speech-to-au) (English-trained deploy packages).

Status: work in progress. Results below are not final and the KPI definition is still to be agreed.

## Data (not included)
- K-MIND_Dyadic SET001-010 and Olkavs (AI Hub), audio only used as input.
- Labels are **OpenFace 2.2.0 pseudo-labels** (AU_r intensity, AU_c occurrence) extracted from the videos, not expert FACS.
- WavLM Base+ features (frozen, resampled to 25 fps) are extracted with `audio_feature_extractor.py`.
- Raw data, labels, features and checkpoints (`*.pt`) are excluded by `.gitignore`.

## Contents
- `train_korean.py` - training/eval: `python train_korean.py <kmind|olkavs|joint> <tcn|mamba|rnn> [scratch|finetune] [epochs] [fold]`
  (env vars: `NSETS`, `AU12_MINPREV`, `SMOOTH`, `LAG_KMIND`, `LAG_OLKAVS`, `OLK_N`, `FEAT`, `TAGX`, ...)
- `extract_feats_ssl.py` - mHuBERT / XLS-R feature extraction (comparison only)
- `model/` - TCN / Mamba / CauseRNN definitions (copied from the AU25/26 deploy package)
- `scripts/` - experiment runners (`run_cv10.sh` = final 10-fold joint run)
- `results/` - per-run metrics CSVs; `logs/` - training logs

Note: `train_korean.py` and `extract_feats_ssl.py` contain absolute Windows paths (`EXP = ...`) and import `model` from
the deploy package path; adjust `EXP` / `sys.path` before running elsewhere.

## Setup (final run)
Joint K-MIND + Olkavs TCN on WavLM features, 10-fold set-level CV on K-MIND (train 8 / val 1 / test 1 sets).
AU25/26 are trained on both datasets; AU12 only on K-MIND (Olkavs AU12 masked). Loss only on own-speech frames.

## Results (10-fold, mean +- sd over test sets)
| Test set | AU | PCC | mF1 (AU_c) | always-active F1 | mF1 (intensity>=0.5) |
|---|---|---|---|---|---|
| K-MIND | AU25 | 0.475 +- 0.053 | 0.611 +- 0.093 | 0.546 | 0.733 |
| K-MIND | AU26 | 0.373 +- 0.066 | 0.490 +- 0.097 | 0.458 | 0.614 |
| K-MIND | AU12 | 0.242 +- 0.196 | 0.373 +- 0.221 | 0.392 | 0.739 |
| Olkavs | AU25 | 0.496 | 0.675 | 0.646 | 0.734 |
| Olkavs | AU26 | 0.427 | 0.358 | 0.510 | 0.642 |

Caveats: AU_c-based mF1 for AU25/26 is below the 60 % KPI; intensity>=0.5 is above it. AU12 is unstable across folds
(speaker label bias). The `*_au12f` runs excluded no speakers and duplicate the plain runs. Olkavs has only 2 test speakers.
