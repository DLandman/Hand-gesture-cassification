# Numbers and figures for the Task 1 section of the report. No model is trained here.
# Run `python analysis.py` with the csv in this folder, figures are saved in work/.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.feature_selection import f_classif

import config
import preprocess as pp

CH = config.CHANNELS
KEY = ["subject", "series"]
WELCH_NPERSEG = 64           # spectrum check, in samples
MIN_SAMPLES_PER_GAP = 3000   # ignore timestamp gaps seen less often than this
config.WORK_DIR.mkdir(exist_ok=True)
# figures are made at the size they have in the report, so the text stays readable
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})

print("=== 1. cleaning")
raw = pp.add_series(pp.read_raw())
clean = pp.load_clean()
print("rows", len(raw), "| subjects", raw.subject.nunique(), "| series", raw.groupby(KEY).ngroups)
same_series = raw[KEY].eq(raw[KEY].shift()).all(axis=1)
repeat = raw[CH + ["class"]].eq(raw[CH + ["class"]].shift()).all(axis=1) & same_series
print(f"rows identical to previous row: {repeat.mean():.2%}")
print("distinct samples:", len(clean), f"({len(clean) / len(raw):.2%} of rows)")

gap = clean.groupby(KEY).time.diff()
seconds = clean.groupby(KEY).time.agg(lambda t: (t.max() - t.min()) / 1000)
rate = clean.groupby(KEY).size() / seconds
print(f"gap between distinct samples (ms): median {gap.median():.0f}, share <= 2 ms {(gap <= 2).mean():.1%}")
print(f"distinct samples per second: mean {rate.mean():.1f}, min {rate.min():.1f}, max {rate.max():.1f}")

# segments = runs of one class inside a series
prev_class = clean.groupby(KEY)["class"].shift()
seg_id = ((clean["class"] != prev_class) | gap.isna()).cumsum()
seg = clean.groupby(seg_id).agg(cls=("class", "first"), subject=("subject", "first"))
n_seg = seg[seg.cls > 0].cls.value_counts().sort_index().to_dict()
print("labeled segments per class:", n_seg, "| class 7 subjects:", sorted(set(int(s) for s in seg[seg.cls == 7].subject)))
big = clean[(gap > config.MAX_GAP_MS) & (clean["class"] == prev_class) & (clean["class"] > 0)]
print(f"gaps > {config.MAX_GAP_MS} ms inside a labeled segment:", len(big))

counts = (clean[CH] / config.COUNT_UNIT).round()
clean["rms"] = np.sqrt((counts ** 2).mean(axis=1))
rms = clean.groupby(["subject", "class"]).rms.mean().unstack()
print("mean RMS by class, median over subjects:", rms.median().round(1).to_dict())
print("class 0 / class 1 RMS ratio per subject: median", round((rms[0] / rms[1]).median(), 1))
lo, hi = config.CLIP_COUNTS
clip = (counts <= lo) | (counts >= hi)
per_subject_clip = clip.any(axis=1).groupby(clean.subject).mean() * 100
print(f"clipped: {clip.values.mean():.3%} of channel values, {clip.any(axis=1).mean():.2%} of samples, "
      f"per subject {per_subject_clip.min():.2f}% to {per_subject_clip.max():.2f}%")

print("\n=== 2. spectrum")
df = pp.label_segments(clean)
df = df[df["class"].isin([0] + config.CLASSES)].copy()
df[CH] = df[CH] / config.COUNT_UNIT
names = {0: "0 unmarked", 1: "1 rest", 2: "2 fist", 3: "3 flexion", 4: "4 extension", 5: "5 radial", 6: "6 ulnar"}

# correlation between a sample and the one before it, per time stamp gap
gest = df[df["class"].isin(config.CLASSES)]
grp = gest.groupby("segment")[CH]
z = (gest[CH] - grp.transform("mean")) / grp.transform("std")      # each channel of each segment: mean 0, std 1
prod = (z * z.groupby(gest.segment).shift()).mean(axis=1)          # mean over the 8 channels
by_gap = prod.groupby(gest.gap_ms).agg(["size", "mean"])
by_gap = by_gap[(by_gap["size"] > MIN_SAMPLES_PER_GAP) & (by_gap.index <= 40)]
print(f"correlation between neighbouring samples, all gaps: {prod.mean():.3f}")
print("correlation per time stamp gap (ms):", by_gap["mean"].round(3).to_dict())

# Welch spectrum per segment, on the sample sequence (no timestamps)
rows = []
for _, s in df.groupby("segment"):
    if len(s) < WELCH_NPERSEG or s.gap_ms.iloc[1:].max() > config.MAX_GAP_MS:
        continue
    freqs, psd = welch(s[CH].to_numpy(), fs=1.0, nperseg=WELCH_NPERSEG, axis=0)
    rows.append((s.subject.iloc[0], s["class"].iloc[0], psd))
tab = pd.DataFrame(rows, columns=["subject", "cls", "psd"])
per_sc = tab.groupby(["subject", "cls"]).psd.apply(lambda s: np.mean(np.stack(s), axis=0))
total = {s: np.mean([per_sc[(s, c)].sum() for c in config.CLASSES]) for s in tab.subject.unique()}
mean_psd = {c: np.mean([v / total[s] for (s, k), v in per_sc.items() if k == c], axis=0).mean(axis=1)
            for c in [0] + config.CLASSES}
print("gesture / rest power ratio:", {names[c]: round(float(mean_psd[c].sum() / mean_psd[1].sum()), 1)
                                      for c in config.CLASSES[1:]})

fig, ax = plt.subplots(1, 3, figsize=(6.7, 2.7))
ax[0].plot(by_gap.index, by_gap["mean"], "o-")
ax[0].axhline(0, color="k", lw=0.5, ls="--")
ax[0].set(title="Neighbour correlation", xlabel="time stamp gap (ms)",
          ylabel="correlation", ylim=(-1, 1))
for c, p in mean_psd.items():
    ax[1].semilogy(freqs, p, color=f"C{c}", label=names[c], ls="--" if c == 0 else "-")
ax[1].set(title="Spectrum by class", xlabel="cycles per sample", ylabel="normalized PSD")
for c in config.CLASSES[1:]:
    ax[2].plot(freqs, mean_psd[c] / mean_psd[1], color=f"C{c}", label=names[c])
ax[2].set(title="Gesture / rest spectrum", xlabel="cycles per sample", ylabel="ratio")
for a in ax:
    a.grid(alpha=0.3)
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, ncol=7, loc="lower center", frameon=False, columnspacing=0.9, handlelength=1.3)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(config.WORK_DIR / "spectrum.png", dpi=250, bbox_inches="tight")

print("\n=== 3. windows")
full = pp.label_segments(clean)
labeled = full[full["class"].isin(config.CLASSES)]
windows, meta, n_cut, n_dropped = pp.cut_windows(labeled)
y, groups = meta["class"].to_numpy(), meta["subject"].to_numpy()
step = int(config.WINDOW_SAMPLES * (1 - config.OVERLAP))
print(f"window {config.WINDOW_SAMPLES} samples, step {step}, shape {windows.shape}")
print("cut:", n_cut, "| dropped for a gap:", n_dropped, "| kept:", len(meta))
print("windows per class:", pd.Series(y).value_counts().sort_index().to_dict())
ps = pd.Series(groups).value_counts()
print("windows per subject: min", ps.min(), "median", int(ps.median()), "max", ps.max())
print("window span in ms (5%, 50%, 95%):", *np.percentile(meta.span_ms, [5, 50, 95]).round())

# check the labels against the full table (class 0 included)
impure = 0
by_series = {k: g for k, g in full.groupby(KEY)}
for (sj, sr), m in meta.groupby(KEY):
    t, c = by_series[(sj, sr)].time.to_numpy(), by_series[(sj, sr)]["class"].to_numpy()
    for t0, span in zip(m.start_time, m.span_ms):
        a, b = np.searchsorted(t, t0), np.searchsorted(t, t0 + span, side="right")
        impure += len(np.unique(c[a:b])) > 1
print("windows with more than one class inside:", impure)


print("\n=== 4. features")
X, cols = pp.build_features(pp.scale_windows(windows, groups, "per_subject"))
ftype = np.array([c.split("_")[0] for c in cols])
corr = np.corrcoef(X, rowvar=False)
for a, b in [("mav", "rms"), ("mav", "wl"), ("rms", "wl"), ("mnf", "mdf")]:
    r = np.mean([corr[cols.index(f"{a}_{c}"), cols.index(f"{b}_{c}")] for c in CH])
    print(f"mean correlation {a} vs {b} (same channel): {r:.2f}")
F = f_classif(X, y)[0]
print("mean F-score by feature type, all windows:",
      pd.Series(F).groupby(ftype).mean().round(0).sort_values(ascending=False).astype(int).to_dict())

print("\n=== 5a. clipping")
freq = pd.Series(counts.to_numpy().ravel().astype(int)).value_counts()
print("count of each value at the top end:   ", {k: int(freq.get(k, 0)) for k in range(hi - 4, hi + 1)})
print("count of each value at the bottom end:", {k: int(freq.get(k, 0)) for k in range(lo, lo + 5)})
top = freq[hi] / np.mean([freq[k] for k in range(hi - 4, hi)])
bottom = freq[lo] / np.mean([freq[k] for k in range(lo + 1, lo + 5)])
print(f"the limit occurs {top:.1f}x (top) and {bottom:.1f}x (bottom) as often as its 4 neighbours")
hit = ((windows <= lo) | (windows >= hi)).any(axis=(1, 2))
print(f"windows with at least one clipped value: {hit.mean():.1%}")

subjects = np.unique(groups)
print("\n=== 5b. amplitude differences between subjects (max / min over subjects)")
std_all = pd.Series({s: windows[groups == s].std() for s in subjects})
std_rest = pd.Series({s: windows[(groups == s) & (y == 1)].std() for s in subjects})
std_ch = np.array([[windows[groups == s][:, :, c].std() for c in range(8)] for s in subjects])
print(f"std of all labeled windows: {std_all.max() / std_all.min():.1f}")
print(f"std of rest windows: {std_rest.max() / std_rest.min():.1f}")
print("std per channel:", (std_ch.max(0) / std_ch.min(0)).round(1).tolist())

print("\n=== 5c. gesture and pause length (the description says 3 s each)")
sg = df.groupby("segment").agg(cls=("class", "first"), subject=("subject", "first"), series=("series", "first"),
                               t0=("time", "first"), t1=("time", "last"), n=("time", "size"))
sg["sec"] = (sg.t1 - sg.t0) / 1000
lab = sg[sg.cls > 0]
print(f"labeled gesture: median {lab.sec.median():.2f} s (5% to 95%: {lab.sec.quantile(.05):.2f} to "
      f"{lab.sec.quantile(.95):.2f}), median {int(lab.n.median())} samples")
between = sg[sg.cls == 0]
pos = between.groupby(["subject", "series"]).cumcount()
rev = between.groupby(["subject", "series"]).cumcount(ascending=False)
print(f"class 0 between two gestures: median {between[(pos > 0) & (rev > 0)].sec.median():.2f} s")
n_gest = lab.groupby(["subject", "series"]).size().mean()
rec = df.groupby(KEY).time.agg(lambda t: (t.max() - t.min()) / 1000).mean()
print(f"recording: mean {rec:.0f} s with {n_gest:.0f} gestures, {n_gest * 6:.0f} s if every gesture and pause took 3 s")
