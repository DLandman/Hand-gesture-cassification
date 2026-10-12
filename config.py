from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "Project_Data_EE4C12_S&S_EMG.csv"
WORK_DIR = ROOT / "work"
CLEAN_FILE = WORK_DIR / "clean_samples.parquet"

SEED = 42

CHANNELS = [f"channel{i}" for i in range(1, 9)]
COUNT_UNIT = 1e-5            # values are int8 counts times this
CLIP_COUNTS = (-128, 127)

# classes used (0 unmarked, 1 rest, 2 fist, 3 flexion, 4 extension, 5 radial, 6 ulnar, 7 palm)
CLASSES = [1, 2, 3, 4, 5, 6]

# merge repeated rows into one sample
COLLAPSE_REPEATS = True
MAX_GAP_MS = 100             # windows with a longer gap between samples are dropped

# windows, in samples
WINDOW_SAMPLES = 20
OVERLAP = 0.5

# normalization: "global" or "per_subject"
NORMALIZATION = "per_subject"

# number of folds for cross-validation split by subject
N_SPLITS = 5

# features
FEATURES = ["mav", "rms", "wl", "zc", "ssc", "mnf", "mdf"]
ZC_THRESHOLD = 0.1

# Choices to sweep later with the real models. The constants above are the defaults.
# Not options (fixed rules): split by subject, fit scalers/selectors on training subjects
# only, windows never mix classes, merge repeated rows, drop windows over a recording gap.
# Also left to the classifier pipeline: feature reduction (none, top k, PCA) and the
# neural network input (raw windows or features).
OPTIONS = {
    "window_samples": [10, 20, 30, 40],
    "overlap": [0, 0.5, 0.75],
    "normalization": ["none", "global", "per_subject"],
    "names": [["rms"], ["mav", "rms", "wl"], ["mav", "rms", "wl", "zc", "ssc"], FEATURES],
    "zc_threshold": [0.05, 0.1, 0.2],
    "drop_clipped": [False, True],
    "classes": [[1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]],
}
