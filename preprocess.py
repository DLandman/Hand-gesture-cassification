# Task 1 pipeline: raw csv -> windows -> per-subject normalization -> features.
# Run `python preprocess.py` to write work/task1_output.npz, or use get_data() from Task 2.
import numpy as np
import pandas as pd

import config

CH = config.CHANNELS


# ---------- load and clean ----------

def read_raw():
    df = pd.read_csv(config.RAW_CSV, dtype={"class": "int8", "label": "int16"})
    return df.rename(columns={"label": "subject"})


def add_series(df):
    # a new series starts where time goes down inside a subject
    new_subject = df["subject"].ne(df["subject"].shift())
    assert df["subject"][new_subject].is_unique, "subjects are not in one block each"
    time_reset = df["time"].lt(df["time"].shift()) & ~new_subject
    series = time_reset.groupby(df["subject"]).cumsum().astype("int8")
    return df.assign(series=series)


def collapse_repeats(df):
    # keep the first row of each run of identical rows (8 channels + class)
    cols = CH + ["class"]
    changed = df[cols].ne(df[cols].shift()).any(axis=1)
    new_series = df[["subject", "series"]].ne(df[["subject", "series"]].shift()).any(axis=1)
    run_start = changed | new_series
    out = df[run_start].copy()
    out["n_rep"] = df.groupby(run_start.cumsum()).size().to_numpy().astype("int32")
    return out.reset_index(drop=True)


def load_clean(force=False):
    # cached in work/ so the big csv is only read once
    if config.CLEAN_FILE.exists() and not force:
        return pd.read_parquet(config.CLEAN_FILE)
    df = add_series(read_raw())
    out = collapse_repeats(df) if config.COLLAPSE_REPEATS else df.assign(n_rep=1)
    config.WORK_DIR.mkdir(exist_ok=True)
    out.to_parquet(config.CLEAN_FILE)
    return out


def label_segments(df):
    # a segment is one run of the same class inside a series
    key = ["subject", "series"]
    gap = df.groupby(key).time.diff()
    prev_class = df.groupby(key)["class"].shift()
    new_segment = (df["class"] != prev_class) | gap.isna()
    return df.assign(segment=new_segment.cumsum(), gap_ms=gap)


# ---------- windows ----------

def cut_windows(df, window_samples=None, overlap=None):
    # windows are cut inside one segment, so they never mix classes
    n = window_samples or config.WINDOW_SAMPLES
    step = max(1, int(n * (1 - (config.OVERLAP if overlap is None else overlap))))
    raw, rows = [], []
    n_cut = n_dropped = 0
    for seg_id, seg in df.groupby("segment"):
        x = (seg[CH].to_numpy() / config.COUNT_UNIT).round().astype("float32")
        t = seg.time.to_numpy()
        gap = seg.gap_ms.to_numpy()
        for start in range(0, len(seg) - n + 1, step):
            n_cut += 1
            if np.nanmax(gap[start + 1:start + n]) > config.MAX_GAP_MS:
                n_dropped += 1
                continue
            raw.append(x[start:start + n])
            rows.append((seg.subject.iloc[0], seg.series.iloc[0], seg_id, seg["class"].iloc[0],
                         t[start], t[start + n - 1] - t[start]))
    cols = ["subject", "series", "segment", "class", "start_time", "span_ms"]
    return np.stack(raw), pd.DataFrame(rows, columns=cols), n_cut, n_dropped


def get_windows(window_samples=None, overlap=None, classes=None):
    # raw windows (n, window_samples, 8) in counts, class, subject id and a meta table
    df = label_segments(load_clean())
    df = df[df["class"].isin(classes or config.CLASSES)]
    raw, meta, _, _ = cut_windows(df, window_samples, overlap)
    return raw, meta["class"].to_numpy(), meta["subject"].to_numpy(), meta


# ---------- normalization ----------

def stats(x):
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(axis=0), flat.std(axis=0)


def scale_windows(raw, groups, mode, fit_idx=None):
    # per channel (x - mean) / std
    if mode == "none":
        return raw.copy()
    if mode == "global":
        if fit_idx is None:
            raise ValueError("global needs fit_idx: the indices of the training windows")
        mu, sd = stats(raw[fit_idx])        # training windows only
        return (raw - mu) / sd
    out = np.empty_like(raw)                # per_subject: each subject's own mean and std
    for s in np.unique(groups):
        mine = groups == s
        mu, sd = stats(raw[mine])
        out[mine] = (raw[mine] - mu) / sd
    return out


# ---------- features ----------

def mav(x):
    return np.abs(x).mean(axis=1)


def rms(x):
    return np.sqrt((x ** 2).mean(axis=1))


def wl(x):
    return np.abs(np.diff(x, axis=1)).sum(axis=1)


def zc(x, threshold):
    # sign changes between neighbours, ignoring tiny ones
    a, b = x[:, :-1], x[:, 1:]
    return ((a * b < 0) & (np.abs(a - b) > threshold)).sum(axis=1)


def ssc(x, threshold):
    # slope sign changes: both neighbours higher or both lower
    d1 = x[:, 1:-1] - x[:, :-2]
    d2 = x[:, 1:-1] - x[:, 2:]
    return (d1 * d2 > threshold).sum(axis=1)


def power_spectrum(x):
    p = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1])         # cycles per sample, 0 to 0.5
    return p, f[None, :, None]


def mnf(x):
    p, f = power_spectrum(x)
    return (p * f).sum(axis=1) / p.sum(axis=1)


def mdf(x):
    # frequency where half of the power is reached
    p, f = power_spectrum(x)
    half = np.cumsum(p, axis=1) >= p.sum(axis=1, keepdims=True) / 2
    return f[0, half.argmax(axis=1), 0]


FEATURE_FUNCTIONS = {"mav": mav, "rms": rms, "wl": wl, "zc": zc, "ssc": ssc, "mnf": mnf, "mdf": mdf}


def build_features(raw, names=None, zc_threshold=None):
    # raw: (n_windows, n_samples, 8) -> X (n_windows, 8 * len(names)) and column names
    names = names or config.FEATURES
    threshold = config.ZC_THRESHOLD if zc_threshold is None else zc_threshold
    x = raw # - raw.mean(axis=1, keepdims=True)     # remove the mean of each window
    parts = [FEATURE_FUNCTIONS[n](x, threshold) if n in ("zc", "ssc") else FEATURE_FUNCTIONS[n](x)
             for n in names]
    return np.hstack(parts), [f"{n}_{ch}" for n in names for ch in CH]


# ---------- output ----------

def make_features(raw, groups, normalization=None, names=None, zc_threshold=None, fit_idx=None):
    # scaled windows and features. "global" needs fit_idx (training windows), so call it inside each fold
    x = scale_windows(raw, groups, normalization or config.NORMALIZATION, fit_idx)
    X, cols = build_features(x, names, zc_threshold)
    return X, x, cols


def get_data(window_samples=None, overlap=None, classes=None, normalization=None, names=None,
             zc_threshold=None, drop_clipped=False, fit_idx=None):
    # X (n, 8 * n_features), y (class), groups (subject id), scaled raw windows (n, window, 8), feature names
    # every argument defaults to the value in config.py; see OPTIONS there for the values to try
    raw, y, groups, _ = get_windows(window_samples, overlap, classes)
    if drop_clipped:
        lo, hi = config.CLIP_COUNTS
        keep = ~((raw <= lo) | (raw >= hi)).any(axis=(1, 2))
        raw, y, groups = raw[keep], y[keep], groups[keep]
    X, x, cols = make_features(raw, groups, normalization, names, zc_threshold, fit_idx)
    return X, y, groups, x, cols


if __name__ == "__main__":
    X, y, groups, raw, cols = get_data()
    np.savez_compressed(config.WORK_DIR / "task1_output.npz", X=X, y=y, groups=groups, raw=raw, columns=cols)
    print("X", X.shape, "| raw windows", raw.shape, "| subjects", len(np.unique(groups)),
          "| windows per class", np.bincount(y)[1:])
