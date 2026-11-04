"""Train Speech->AU (AU12/25/26) on Korean data with OpenFace pseudo-labels; evaluate with mF1.
usage: python -I train_korean.py <kmind|olkavs> <tcn|mamba|rnn> [scratch|finetune] [epochs]
kmind : train SET001-003, val SET004, test SET005; loss only on own-only speech frames (VAD json)
olkavs: train speaker C920 (takes 1-10), val C920 takes 11-12, test speaker C221; loss only on speech sentences."""
import csv, glob, json, os, re, sys, tarfile, zipfile
import numpy as np, torch, torch.nn as nn
from scipy.ndimage import gaussian_filter1d
EXP = r"C:\Users\user\Documents\IITP-29\experiment"
sys.path.insert(0, os.path.join(EXP, "lipsync_7au_deploy", "lipsync_au25_26_17_deploy"))
from model import SpeechToAU, SpeechToAUMamba, CauseRNN
DS, ARCH = sys.argv[1], sys.argv[2]
INIT = sys.argv[3] if len(sys.argv) > 3 else "scratch"
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
NSETS = int(os.environ.get("NSETS", 10)); AU12_MINPREV = float(os.environ.get("AU12_MINPREV", 0))  # drop K-MIND speakers with AU12 AU_c prevalence below this from AU12 training loss
SMOOTH = float(os.environ.get("SMOOTH", 0))  # gaussian sigma (frames) applied to TRAINING labels only
TAGX = os.environ.get("TAGX", "")
CLEAN_ROT = bool(os.environ.get("CLEAN_ROT")); INCL_MIX = bool(os.environ.get("INCL_MIX"))  # SET003-5 rotation; add SET001/002 (MIX audio) to train
FEAT = os.environ.get("FEAT", "")  # "" = WavLM Base+ (korean_features), else korean_features_<FEAT>
OLK_N = int(os.environ["OLK_N"]) if os.environ.get("OLK_N") else None  # limit Olkavs train takes (data-amount curve)
FOLD = int(sys.argv[5]) if len(sys.argv) > 5 else None  # K-MIND set-level CV: test=SET00(FOLD+1), val=next set
AUS = ["AU12", "AU25", "AU26"]; dev = "cuda" if torch.cuda.is_available() else "cpu"; FR = 25; WIN = 400
torch.manual_seed(0); np.random.seed(0)


def T(x):
    h, m, s = x.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)


def load_of(p):
    rs = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(p))]
    t = np.array([float(r["timestamp"]) for r in rs])
    return t, {a: np.array([float(r[a + "_r"]) for r in rs]) for a in AUS}, {a: np.array([float(r[a + "_c"]) for r in rs]) for a in AUS}


def ivmask(n, ivs):
    m = np.zeros(n, bool)
    for a, b in ivs: m[max(0, int(round(a * FR))):int(round(b * FR))] = True
    return m


def build(stem, ds, featp, mask_fn):
    f = np.load(featp); t, r, c = load_of(os.path.join(EXP, "korean_au_labels", ds, stem, stem + ".csv"))
    n = min(len(f), int(t[-1] * FR) + 1); lag = float(os.environ.get(f"LAG_{ds.upper()}", 0)); tt = np.arange(n) / FR + lag  # y_new[t] = y(t + lag)
    y = np.stack([np.interp(tt, t, r[a]) for a in AUS], 1).astype(np.float32)
    u = dict(stem=stem, f=f[:n], y=y, c=np.stack([np.interp(tt, t, c[a]) >= 0.5 for a in AUS], 1), m=mask_fn(n))
    if SMOOTH > 0: u["ys"] = gaussian_filter1d(y, SMOOTH, axis=0, mode="nearest").astype(np.float32)
    return u


def load(DS_):
    units = {"train": [], "val": [], "test": []}
    if DS_ == "kmind":
        KB = os.path.join(EXP, "dataset", "K-MIND_Dyadic", "Dataset", "Opendatset"); FD = os.path.join(EXP, "korean_features" + ("_" + FEAT if FEAT else ""), "kmind")
        for zp in sorted(glob.glob(os.path.join(KB, "*.zip"))):
            z = zipfile.ZipFile(zp); d = json.loads(z.read([i for i in z.infolist() if i.filename.endswith(".json")][0]))
            sets = {s["subset_id"]: s for s in d["subset_info"]["sets"]}
            for stem in sorted(os.path.basename(i.filename)[:-4] for i in z.infolist() if i.filename.endswith("_CS.mp4")):
                m = re.match(r"(SET\d+_M\d+_S\d+)_(\d{3})_", stem); sid, spk = m.group(1), m.group(2); oth = "002" if spk == "001" else "001"
                sec = sets[sid]["dialogue"]["section"]
                def iv(k): return [(T(x["start_time"]), T(x["end_time"])) for x in sec if x["speaker_id"] == k and x["vocal_or_body"] == "vocal"]
                fp = os.path.join(FD, stem[:-3] + "_P.npy")
                if not os.path.isfile(fp): fp = glob.glob(os.path.join(FD, sid + "_MIX_*_P.npy"))[0]
                u = build(stem, "kmind", fp, lambda n: ivmask(n, iv(spk)) & ~ivmask(n, iv(oth)))
                sp = stem[:6]
                si = int(sp[3:]) - 1
                if CLEAN_ROT:
                    cl = [2, 3, 4]; te, va = cl[FOLD], cl[(FOLD + 1) % 3]
                    if si == te: units["test"].append(u)
                    elif si == va: units["val"].append(u)
                    elif si in cl or INCL_MIX: units["train"].append(u)
                elif FOLD is None: units["train" if si < 3 else "val" if si == 3 else "test"].append(u)
                else: units["test" if si == FOLD else "val" if si == (FOLD + 1) % NSETS else "train"].append(u)
    else:
        OB = os.path.join(EXP, "dataset", "Olkavs", "Dataset"); FD = os.path.join(EXP, "korean_features" + ("_" + FEAT if FEAT else ""), "olkavs"); js = {}
        for tp in (os.path.join(OB, "Training", "라벨링데이터", "TL174.tar"), os.path.join(OB, "Validation", "라벨링데이터", "VL11.tar"),
                   os.path.join(OB, "Training", "라벨링데이터", "TL181.tar"), os.path.join(OB, "Training", "라벨링데이터", "TL219.tar"), os.path.join(OB, "Validation", "라벨링데이터", "VL7.tar")):
            with tarfile.open(tp) as t:
                for mm in t:
                    if re.search(r"_A_\d{3}\.json$", mm.name): js[os.path.basename(mm.name)[:-5]] = json.load(t.extractfile(mm))[0]
        for fp in sorted(glob.glob(os.path.join(FD, "*.npy"))):
            stem = os.path.basename(fp)[:-4]; ivs = [(x["start_time"], x["end_time"]) for x in js[stem]["Sentence_info"]]
            u = build(stem, "olkavs", fp, lambda n: ivmask(n, ivs)); k = int(stem[-3:])
            u["spk"] = re.search(r"_([CE]\d+)_A_", stem).group(1)
            if u["spk"] in ("C221", "E074"): units["test"].append(u)
            elif u["spk"] == "E220": units["val"].append(u)
            elif OLK_N is None or k <= OLK_N: units["train"].append(u)
    for us in units.values():
        for u in us:
            u["ds"] = DS_; u["am"] = np.array([DS_ == "kmind", True, True])
            if DS_ == "kmind": u["spk"] = re.match(r"(SET\d+_M\d+)_S\d+_(\d{3})_", u["stem"]).expand(r"_")
    return units

units = {k: [] for k in ("train", "val", "test")}
for d_ in (["kmind", "olkavs"] if DS == "joint" else [DS]):
    for k, v in load(d_).items(): units[k] += v
for k, v in units.items():
    print(k, len(v), "units", sum(len(u["f"]) for u in v), "frames", sum(int(u["m"].sum()) for u in v), "masked-in", flush=True)

tr = units["train"]
if AU12_MINPREV > 0:
    sp_c = {}
    for u in tr:
        if u["ds"] == "kmind": sp_c.setdefault(u["spk"], []).append(u["c"][u["m"], 0])
    low = {k for k, v in sp_c.items() if np.concatenate(v).mean() < AU12_MINPREV}
    for u in tr:
        if u["ds"] == "kmind" and u["spk"] in low: u["am"] = np.array([False, True, True])
    print("AU12 excluded speakers (train, AU_c prev <", AU12_MINPREV, "):", sorted(low))
mean = np.concatenate([u["f"][u["m"]] for u in tr]).astype(np.float32).mean(0)
std = np.concatenate([u["f"][u["m"]] for u in tr]).astype(np.float32).std(0) + 1e-6
ymean = np.zeros(3, np.float32); ystd = np.ones(3, np.float32)
for i in range(3):
    ys_ = [u["y"][u["m"], i] for u in tr if u["am"][i]]
    if ys_: yi = np.concatenate(ys_); ymean[i], ystd[i] = yi.mean(), yi.std() + 1e-6
print("label mean", ymean.round(2), "std", ystd.round(2))
FAC = {"tcn": SpeechToAU, "mamba": SpeechToAUMamba, "rnn": CauseRNN}[ARCH]
FDIM = tr[0]["f"].shape[1]
model = FAC(input_dim=FDIM, num_aus=len(AUS)).to(dev)
if INIT == "finetune":  # English checkpoint, shape-matching tensors only (output head re-initialised for the 3 Korean targets)
    src = {"tcn": "grid_speechau_v1_1/final_specialist_checkpoints/tcn_wavlm.pt", "mamba": "grid_speechau_v1_1/final_specialist_checkpoints/mamba_wavlm.pt",
           "rnn": "cremad_speechau_v1/final_specialist_checkpoints/cause_rnn_wavlm.pt"}[ARCH]
    ck = torch.load(os.path.join(EXP, src), map_location="cpu", weights_only=False); sd = model.state_dict()
    ok = {k: v for k, v in ck["model_state"].items() if k in sd and v.shape == sd[k].shape}; sd.update(ok); model.load_state_dict(sd)
    print("finetune: loaded", len(ok), "/", len(sd), "tensors")


def prep(u): return torch.from_numpy((u["f"].astype(np.float32) - mean) / std)


def predict(u):
    x = prep(u); out = []
    for s in range(0, len(x), WIN):
        with torch.no_grad(): out.append(model(x[s:s + WIN].unsqueeze(0).to(dev)).squeeze(0).cpu().numpy())
    return np.concatenate(out) * ystd + ymean


def pcc(a, b): return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-9 and b.std() > 1e-9 else float("nan")


def f1(yt, yp):
    tp = int((yt & yp).sum()); d = 2 * tp + int((~yt & yp).sum()) + int((yt & ~yp).sum()); return 2 * tp / d if d else float("nan")


def collect(us):
    P = [predict(u) for u in us]
    return np.concatenate([p[u["m"]] for p, u in zip(P, us)]), np.concatenate([u["y"][u["m"]] for u in us]), np.concatenate([u["c"][u["m"]] for u in us])


groups = sorted({u["ds"] for u in tr}); valid = {g: [any(bool(u["am"][i]) for u in tr if u["ds"] == g) for i in range(3)] for g in groups}
valid_all = {g: [True, True, True] if g == "kmind" else valid[g] for g in groups}
opt = torch.optim.AdamW(model.parameters(), lr=3e-4 if INIT == "scratch" else 1e-4, weight_decay=1e-2)
best, bad, tag = -9, 0, f"{DS}_{ARCH}_{INIT}" + (f"_fold{FOLD}" if FOLD is not None else "") + (f"_olkN{OLK_N}" if OLK_N else "") + (f"_feat{FEAT}" if FEAT else "") + TAGX
ck_path = os.path.join(EXP, "korean_train", tag + ".pt"); w = np.array([len(u["f"]) for u in tr], float); w /= w.sum()
if os.environ.get("PROBE_ONLY"):  # audio/video lag: pair prediction at t with label at t+L, pooled over test units (own-speech frames only)
    model.load_state_dict(torch.load(ck_path)); model.eval(); PR = [(u, predict(u)) for u in units["test"]]; Ls = list(range(-10, 11)); curve = {}
    for L in Ls:
        for i in range(3):
            a_, b_ = [], []
            for u, P in PR:
                T_ = len(P); t0, t1 = max(0, -L), min(T_, T_ - L); idx = np.arange(t0, t1); ok = u["m"][idx] & u["m"][idx + L]
                a_.append(P[idx[ok], i]); b_.append(u["y"][idx[ok] + L, i])
            curve[(L, i)] = pcc(np.concatenate(a_), np.concatenate(b_))
    for i, a in enumerate(AUS):
        c_ = np.array([curve[(L, i)] for L in Ls]); print(a, "pooled PCC by lag(frames=40ms):", " ".join(f"{L}:{v:.3f}" for L, v in zip(Ls, c_)), "| best", Ls[int(np.argmax(c_))])
    c2 = np.array([(curve[(L, 1)] + curve[(L, 2)]) / 2 for L in Ls]); k = int(np.argmax(c2)); fr = 0.0
    if 0 < k < len(Ls) - 1: d_ = c2[k - 1] - 2 * c2[k] + c2[k + 1]; fr = 0.5 * (c2[k - 1] - c2[k + 1]) / d_ if d_ else 0.0
    print(f"AU25/26 pooled best lag = {(Ls[k] + fr) * 40:.0f} ms (PCC {c2[k]:.3f} vs {c2[Ls.index(0)]:.3f} at 0)")
    pv = []
    for u, P in PR:
        if u["m"].sum() < 2000: continue
        sc = []
        for L in Ls:
            T_ = len(P); idx = np.arange(max(0, -L), min(T_, T_ - L)); ok = u["m"][idx] & u["m"][idx + L]
            sc.append(np.nanmean([pcc(P[idx[ok], i], u["y"][idx[ok] + L, i]) for i in (1, 2)]))
        pv.append(Ls[int(np.argmax(sc))] * 40)
    print("per-video best lag ms: median", np.median(pv), "IQR", np.percentile(pv, [25, 75]), "all", pv); sys.exit(0)
for ep in range(EPOCHS):
    model.train(); tl = 0; nb = 0
    for _ in range(max(40, len(tr) * 3)):
        xs, ys, ms = [], [], []
        for _ in range(8):
            u = tr[np.random.choice(len(tr), p=w)]; s = np.random.randint(0, max(1, len(u["f"]) - WIN)); sl = slice(s, s + WIN)
            x = prep(u)[sl]; y = torch.from_numpy((u.get("ys", u["y"])[sl] - ymean) / ystd); m = torch.from_numpy(u["m"][sl, None] & u["am"][None, :]); pad = WIN - len(x)
            if pad > 0: x = nn.functional.pad(x, (0, 0, 0, pad)); y = nn.functional.pad(y, (0, 0, 0, pad)); m = nn.functional.pad(m, (0, 0, 0, pad))
            xs.append(x); ys.append(y); ms.append(m)
        x, y, m = torch.stack(xs).to(dev), torch.stack(ys).to(dev), torch.stack(ms).to(dev)
        loss = (((model(x) - y) ** 2) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item(); nb += 1
    model.eval(); vps = []
    for g in groups:
        Pv, Yv, _ = collect([u for u in units["val"] if u["ds"] == g]); vps.append(float(np.nanmean([pcc(Pv[:, i], Yv[:, i]) for i in range(3) if valid[g][i]])))
    vp = float(np.mean(vps))
    print(f"ep{ep + 1} train_loss {tl / nb:.3f} val_pcc {vp:.3f}", flush=True)
    if vp > best: best, bad = vp, 0; torch.save(model.state_dict(), ck_path)
    else:
        bad += 1
        if bad >= 6: break
model.load_state_dict(torch.load(ck_path)); model.eval()
rows = []
for g in groups:
    Pv, Yv, Cv = collect([u for u in units["val"] if u["ds"] == g])
    subsets = [(g, lambda u: True)] + ([("olkavs_C221", lambda u: u["spk"] == "C221"), ("olkavs_E074", lambda u: u["spk"] == "E074")] if g == "olkavs" else [])
    for gname, filt in subsets:
        Pt, Yt, Ct = collect([u for u in units["test"] if u["ds"] == g and filt(u)]); gr = []
        for i, a in enumerate(AUS):
            if not valid_all[g][i]: continue
            # occurrence definition: truth = OpenFace AU_c; prediction threshold fit on val for best F1
            cand = np.quantile(Pv[:, i], np.linspace(0.05, 0.95, 37)); th = max(cand, key=lambda t: np.nan_to_num(f1(Cv[:, i], Pv[:, i] >= t)))
            pr = float(Ct[:, i].mean()); fc = f1(Ct[:, i], Pt[:, i] >= th)
            r = dict(dataset=gname, run=tag, au=a, n_test=len(Yt), pcc=round(pcc(Pt[:, i], Yt[:, i]), 3), mae=round(float(np.abs(Pt[:, i] - Yt[:, i]).mean()), 3),
                     prev_c=round(pr, 3), f1_c=round(fc, 3), macroF1_c=round((fc + f1(~Ct[:, i], Pt[:, i] < th)) / 2, 3), always_active_f1_c=round(2 * pr / (1 + pr), 3))
            for t in (0.5, 1.0, 2.0): r[f"f1_t{t}"] = round(f1(Yt[:, i] >= t, Pt[:, i] >= t), 3); r[f"pos_t{t}"] = round(float((Yt[:, i] >= t).mean()), 3)
            gr.append(r); print(r, flush=True)
        print(f"[{gname}] mF1 (AU_c, {len(gr)} AUs) = {np.nanmean([r['f1_c'] for r in gr]):.3f}   macro-F1 = {np.nanmean([r['macroF1_c'] for r in gr]):.3f}   always-active = {np.mean([r['always_active_f1_c'] for r in gr]):.3f}")
        for t in (0.5, 1.0, 2.0): print(f"[{gname}] mF1 (intensity>={t}) = {np.nanmean([r[f'f1_t{t}'] for r in gr]):.3f}")
        rows += gr
with open(os.path.join(EXP, "korean_train", f"results_{tag}.csv"), "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
