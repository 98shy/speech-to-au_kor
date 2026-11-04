# speech-to-au_kor / deploy

Korean-trained speech -> facial Action Unit (AU) intensity inference: **AU12, AU25, AU26** at 25 fps.
Self-contained, same interface as the English packages in [speech-to-au](https://github.com/98shy/speech-to-au).

## Quick start

```bash
pip install -r requirements.txt
python infer.py --wav path/to/speech.wav --out predictions.npz
```

```python
import infer
result = infer.predict("path/to/speech.wav")
# {"timestamp_sec": [...], "AU12": [...], "AU25": [...], "AU26": [...]}
```

Each AU array is float32, clipped to OpenFace's [0, 5] intensity scale, one value per 25 fps frame.
The first call downloads WavLM Base+ (`microsoft/wavlm-base-plus`, pinned revision) from the Hugging Face Hub.

## What it is
- Feature: WavLM Base+ final hidden state (768-d), frozen, resampled to 25 fps (`audio_feature_extractor.py`, identical to the English packages).
- Model: TCN (`model/speech_to_au_tcn.py`), 3 outputs (AU12, AU25, AU26). `checkpoints/kor_tcn_fold0..9.pt` are the
  10 models of a 10-fold set-level cross-validation on K-MIND; `infer.py` **averages their predictions**.
  Every checkpoint stores its own feature `mean`/`std` and output scaling `ymean`/`ystd`, fit on that fold's training set.
- Training data: K-MIND_Dyadic SET001-010 (AU12, AU25, AU26) + Olkavs (AU25, AU26 only; AU12 masked), jointly.
  Labels are **OpenFace 2.2.0 pseudo-labels**, not expert FACS. Loss is computed on the speaker's own speech frames only.

## Expected quality (single fold models, held-out K-MIND sets, own-speech frames; mean +- sd over 10 folds)
| AU | PCC | mF1 (AU_c occurrence) | mF1 (intensity >= 0.5) |
|---|---|---|---|
| AU25 | 0.475 +- 0.053 | 0.611 +- 0.093 | 0.733 |
| AU26 | 0.373 +- 0.066 | 0.490 +- 0.097 | 0.614 |
| AU12 | 0.242 +- 0.196 | 0.373 +- 0.221 | 0.739 |

The averaged ensemble in `infer.py` has **not** been evaluated on held-out data: each fold model was held out on a
different set, so every K-MIND set was seen in training by 9 of the 10 models. The table is for the single models.

## Limitations - read before using for rendering
- **Not a lip-sync ground truth.** The targets are OpenFace estimates. They carry OpenFace's own errors and speaker-specific bias.
- **AU12 is unreliable.** Its PCC ranges from -0.13 to 0.47 across folds (speaker label bias); treat it as weak. AU25/AU26 are moderate (PCC about 0.4-0.5).
- **Own speech only.** Training and evaluation used only frames where the person is speaking. Silence and listening
  segments were never supervised, so outputs there are not meaningful. On a recording that includes the partner's
  speech, agreement with OpenFace over all frames is much lower (e.g. PCC 0.2-0.28 on one SET005 file).
- Output intensities are compressed toward the mean (regression to the mean); expect smaller range than OpenFace.
- The unit/scale is OpenFace AU_r, not mocap distances or blendshape weights; map to your rig accordingly.

## Data terms
The checkpoints are derived from AI Hub datasets (K-MIND_Dyadic, Olkavs). Check their terms of use before redistributing
or using the models outside the project.

## Sanity check
Features from `audio_feature_extractor.py` on a K-MIND wav match the stored training features (frame-wise correlation 0.9994),
and predictions from fresh vs stored features correlate at 0.9996 (max difference 0.12 intensity). Each exported fold model
reproduces the PCC of its original evaluation run on its test set.
