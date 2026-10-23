# Hand gesture classification from EMG signals (EE4C12)

Code for the EE4C12 project, classifying six hand gesture classes from 8-channel EMG signals of 36 subjects. Task 1 is the pre-processing, Tasks 2 and 3 are the models and the ensemble.

## Setup

Install the packages with `pip install -r requirements.txt`. The data file `Project_Data_EE4C12_S&S_EMG.csv` is not included in the repository, put it in this folder.

## Task 1, pre-processing

- `preprocess.py` turns the csv into windows and features. The first run takes about 10 seconds and saves a cleaned copy of the data in `work/`. `python preprocess.py` saves the result in `work/task1_output.npz`, and `get_data()` returns it to other scripts.
- `analysis.py` prints the numbers used in the Task 1 section of the report and saves the spectrum figure in `work/`. It takes about 20 seconds.
- `config.py` holds the settings (window length, overlap, normalization, features).

`get_data()` returns `X, y, groups, windows, names`.

| | shape | what |
|---|---|---|
| `X` | (13335, 56) | features, one row per window |
| `y` | (13335,) | gesture class 1 to 6 |
| `groups` | (13335,) | subject number, used to split by subject |
| `windows` | (13335, 40, 8) | the standardized windows (40 samples, 8 channels) |
| `names` | 56 | feature names, for example `rms_channel3` |

The steps are described in the Task 1 section of the report. In short, repeated rows are merged, only classes 1 to 6 are kept, windows of 40 samples with 75% overlap are cut inside one gesture, every subject is standardized separately, and 7 features are computed per channel.

## Tasks 2 and 3, models

`main.ipynb` contains the baseline models, the pre-processing comparison, the feature reduction, the hyperparameter tuning, the ensemble and the final test on 8 subjects that were kept apart from the start. The hyperparameter searches take hours, so their results are kept in the saved outputs of the notebook.

## Notes

- All splits between training and test data are made by subject, never by window.
- The sampling rate of the data is not given, so windows are counted in samples.
