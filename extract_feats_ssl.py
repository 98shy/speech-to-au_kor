"""usage: extract_feats_ssl.py <hf_model_id> <layer> <out_name>. Same wavs/pipeline as extract_feats.py (mono, 16 kHz, 20 s chunks, linear interpolation to 25 fps), hidden_states[layer], float16."""
import glob, io, math, os, re, sys, tarfile, time, zipfile
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel
MID, LAYER, NAME = sys.argv[1], int(sys.argv[2]), sys.argv[3]
EXP = r"C:\Users\user\Documents\IITP-29\experiment"; OUT = os.path.join(EXP, "korean_features_" + NAME)
sys.path.insert(0, os.path.join(EXP, "lipsync_7au_deploy", "lipsync_au25_26_17_deploy"))
import audio_feature_extractor as afe
dev = "cuda" if torch.cuda.is_available() else "cpu"
fe = AutoFeatureExtractor.from_pretrained(MID); model = AutoModel.from_pretrained(MID).to(dev).eval()
print("model", MID, "layers", model.config.num_hidden_layers, "hidden", model.config.hidden_size, "use layer", LAYER, "normalize", getattr(fe, "do_normalize", None), flush=True)

def feat(data):
    x, sr = sf.read(io.BytesIO(data)); x = x.mean(1) if x.ndim == 2 else x
    g = math.gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    outs = []
    for i in range(0, len(x), 16000 * 20):
        inp = fe(x[i:i + 320000], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad(): h = model(inp.input_values.to(dev), output_hidden_states=True).hidden_states[LAYER]
        outs.append(h.squeeze(0).cpu().numpy().astype(np.float32))
    return afe.resample_to_fps(np.concatenate(outs), 320 / 16000, 25)

def save(ds, stem, data):
    p = os.path.join(OUT, ds, stem + ".npy")
    if os.path.isfile(p): return
    os.makedirs(os.path.dirname(p), exist_ok=True); t0 = time.time(); f = feat(data); np.save(p, f.astype(np.float16)); print("done", ds, stem, f.shape, f"{time.time()-t0:.0f}s", flush=True)

KB = os.path.join(EXP, "dataset", "K-MIND_Dyadic", "Dataset", "Opendatset")
for zp in sorted(glob.glob(os.path.join(KB, "*.zip"))):
    with zipfile.ZipFile(zp) as z:
        for i in z.infolist():
            if i.filename.endswith(".wav"): save("kmind", os.path.basename(i.filename)[:-4], z.read(i))
OB = os.path.join(EXP, "dataset", "Olkavs", "Dataset")
for tp in (os.path.join(OB, "Training", "원천데이터", "TS174.tar"), os.path.join(OB, "Validation", "원천데이터", "VS11.tar")):
    with tarfile.open(tp) as t:
        for m in t:
            if re.search(r"_A_\d{3}\.wav$", m.name): save("olkavs", os.path.basename(m.name)[:-4], t.extractfile(m).read())
print("ALL DONE", flush=True)
