# EMG project: data pre-processing (Task 1)

This turns the raw csv into windows and features that are ready for the models.

## Setup

Put `Project_Data_EE4C12_S&S_EMG.csv` in this folder (it is not in git). You need `pandas`, `numpy` and `pyarrow`.
The first run takes about 30 seconds and saves a cleaned copy in `work/`, after that it is fast.

## Use

```python
from preprocess import get_data
X, y, groups, raw, cols = get_data()
```

| | shape | what |
|---|---|---|
| `X` | (15065, 56) | features, one row per window |
| `y` | (15065,) | class 1 to 6 |
| `groups` | (15065,) | subject id (1 to 36), use it to split by subject |
| `raw` | (15065, 20, 8) | the windows themselves (20 samples, 8 channels), for the neural network |
| `cols` | 56 names | e.g. `rms_channel3` |

`python preprocess.py` saves the same arrays in `work/task1_output.npz`.

## What it does

1. Rows that repeat the previous row exactly are merged (89% of the csv), because the sensor updates slower than the clock that wrote the rows.
2. Only classes 1 to 6 are used (class 0 is unmarked, class 7 exists for 2 subjects only).
3. Windows of 20 samples, 50% overlap, always inside one gesture segment. Windows over a recording gap are dropped.
4. Each channel is standardized per subject (own mean and std). `raw` is this standardized version.
5. Per window and channel: MAV, RMS, WL, ZC, SSC, MNF, MDF, so 7 x 8 = 56 features. Columns are in blocks of 8 channels per feature type.

## Options

Defaults are in `config.py`. `get_data(...)` takes the same things as arguments: `window_samples`, `overlap`, `classes`, `normalization`, `names` (feature types), `zc_threshold`, `drop_clipped`. `OPTIONS` in `config.py` lists the values we want to try, e.g. `get_data(window_samples=30, names=["rms"])`.

`normalization="global"` has to be fit on the training subjects only, so `get_data` refuses it. Inside each fold use
`make_features(raw_counts, groups, "global", fit_idx=train)` (get the raw counts from `get_windows()`).

## Rules I suggest we keep

- Split by subject (`GroupKFold` on `groups`), never by window. Neighbouring windows are near copies.
- Fit scalers, feature selection and PCA on the training subjects only.
- Keep a few subjects (say 6 of 36) locked away and only use them for the final score, so the sweep does not tune on the test set.

## Things to know

- A quick logistic regression gave about 0.86 accuracy on held-out subjects with per-subject scaling (0.80 with global), chance is 0.17. (Just a sanity check)
- One feature per channel (RMS) did as well as all 56 in that check, so the extra features may not matter. Worth testing with a better model. 
- Some subjects are much harder than others (the worst gets about 0.3), so look at scores per subject.
- Per-subject scaling needs a bit of data from a new person before predicting. I want to check with a TA that this is fine.
- Class 0 ("no gesture") is not included. Adding it naively makes the model predict class 0 almost always (65% of windows are class 0), and its not really what is asked from us so Id suggest just dropping it.
- The sampling rate in Hz is not given. I measured about 105 distinct samples per second by timestamp, and the timestamps are not a reliable clock, which is why windows are counted in samples. 20 samples is about 180 ms by timestamp.
- `drop_clipped=True` removes 12% of the windows (sensor limit hit), so by default it is off, since the clipping didnt seem like that big of an issue, but might be worth testing.
- Window length and the ZC threshold are untuned starting values.
