import numpy as np
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
WINDOW_SAMPLES = 40
OVERLAP = 0.75

# normalization: "global" or "per_subject"
NORMALIZATION = "per_subject"

# number of folds for cross-validation split by subject
N_SPLITS = 5

# features
FEATURES = ["mav", "rms", "wl", "zc", "ssc", "mnf", "mdf"]
ZC_THRESHOLD = 0.1

# toggle custom names for pipeline steps
CUSTOM_NAMES = False

# scoring metrics for hyperparameter tuning
SCORING = None
# SCORING = ["accuracy", "recall_macro", "precision_macro", "f1_macro"]

# hyperparameter options
uniform_spacer = lambda x, y: (pow(10, x), pow(10, y), int(np.log10(pow(10, y) / pow(10, x))) + 1)
C_OPTS = np.geomspace(*uniform_spacer(-3, 3), dtype=float).tolist()
GAMMA_OPTS = ['scale', 'auto'] #+ np.geomspace(*uniform_spacer(-3, -1), dtype=float).tolist()
COEF0_OPTS = [0, 1, 5]
ALPHA_OPTS = np.geomspace(*uniform_spacer(-4, -1), dtype=float).tolist()
HLS_OPTS = [(100,), (50, 50), (100, 50), (100, 100), (50, 50, 50), (100, 50, 25), (100, 100, 100)]
ACTIVATION_OPTS = ['identity', 'logistic', 'tanh', 'relu']
LR_INIT_OPTS = np.geomspace(*uniform_spacer(-4, -1), dtype=float).tolist()
MOMENTUM_OPTS = [0.5, 0.9, 0.99]
N_ESTIMATORS_OPTS = [10, 25, 50, 100]
MAX_DEPTH_OPTS = [None, 5, 10, 20]
MIN_SAMPLES_SPLIT_OPTS = [2, 3, 4]
MAX_FEATURES_OPTS = ['sqrt', 'log2', None, 0.3, 0.5, 0.7]
CCP_ALPHA_OPTS = [0.0, 0.01, 0.05, 0.1]

# Choices to sweep later with the real models. The constants above are the defaults.
# Not options (fixed rules): split by subject, fit scalers/selectors on training subjects
# only, windows never mix classes, merge repeated rows, drop windows over a recording gap.
# Also left to the classifier pipeline: feature reduction (none, top k, PCA) and the
# neural network input (raw windows or features).
OPTIONS = {
    'preprocessing': {
        "window_samples": [10, 20, 30, 40],
        "overlap": [0, 0.25, 0.5, 0.75],
        "normalization": ["none", "global", "per_subject"],
        "names": [FEATURES[1], FEATURES[:3], FEATURES[:5], FEATURES],
        "zc_threshold": [0.05, 0.1, 0.2],
        "drop_clipped": [False, True],
        "classes": [[1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]],
    },
    'pca': {
        'n_components': [2, 5, 10, None],
    },
    'lineardiscriminantanalysis': [
        {
            'solver': ['svd'],
        }, {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': [None, 'auto'],
        }
    ],
    'randomforestclassifier': [
        {
            'n_estimators': N_ESTIMATORS_OPTS,
            'max_depth': MAX_DEPTH_OPTS,
            'min_samples_split': MIN_SAMPLES_SPLIT_OPTS,
            'max_features': MAX_FEATURES_OPTS,
            'bootstrap': [True],
            'oob_score': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'ccp_alpha': CCP_ALPHA_OPTS,
        }, {
            'n_estimators': N_ESTIMATORS_OPTS,
            'max_depth': MAX_DEPTH_OPTS,
            'min_samples_split': MIN_SAMPLES_SPLIT_OPTS,
            'max_features': MAX_FEATURES_OPTS,
            'bootstrap': [False],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'ccp_alpha': CCP_ALPHA_OPTS,
        },
    ],
    'votingclassifier': {
            'voting': ['soft', 'hard'],
    },
    'selectkbest': {
        'k': [2, 5, 10, "all"],
    },
    'logisticregression': [
        {
            'C': C_OPTS,
            'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
            'class_weight': [None, 'balanced'],
            'solver': ['saga'],
        }, {
            'C': [np.inf],
            'class_weight': [None, 'balanced'],
            'solver': ['saga'],
        }, {
            'C': C_OPTS + [np.inf],
            'class_weight': [None, 'balanced'],
            'solver': ['lbfgs', 'newton-cg', 'sag'],
        }
    ],
    'kneighborsclassifier': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance', None],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 1.5, 2, 3],
    },
    'mlpclassifier': [
        {
            'hidden_layer_sizes': HLS_OPTS,
            'activation': ACTIVATION_OPTS,
            'solver': ['lbfgs'],
            'alpha': ALPHA_OPTS,
        }, {
            'hidden_layer_sizes': HLS_OPTS,
            'activation': ACTIVATION_OPTS,
            'solver': ['sgd'],
            'alpha': ALPHA_OPTS,
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': LR_INIT_OPTS,
            'momentum': MOMENTUM_OPTS,
        }, {
            'hidden_layer_sizes': HLS_OPTS,
            'activation': ACTIVATION_OPTS,
            'solver': ['sgd'],
            'alpha': ALPHA_OPTS,
            'learning_rate': ['invscaling'],
            'learning_rate_init': LR_INIT_OPTS,
            'power_t': [0.25, 0.5, 1.0, 2.0],
            'momentum': MOMENTUM_OPTS,
        }, {
            'hidden_layer_sizes': HLS_OPTS,
            'activation': ACTIVATION_OPTS,
            'solver': ['adam'],
            'alpha': ALPHA_OPTS,
            'learning_rate_init': LR_INIT_OPTS,
            'beta_1': [0.9, 0.99, 0.999],
            'beta_2': [0.9, 0.99, 0.999],
            'epsilon': np.geomspace(*uniform_spacer(-8, -4), dtype=float).tolist(),
        }
    ],
    'svc': [
        {
            'C': C_OPTS,
            'kernel': ['linear'],
            'shrinking': [True, False],
            'class_weight': [None, 'balanced'],
            'break_ties': [True, False],
        }, {
            'C': C_OPTS,
            'kernel': ['rbf'],
            'gamma': GAMMA_OPTS,
            'shrinking': [True, False],
            'class_weight': [None, 'balanced'],
            'break_ties': [True, False],
        }, {
            'C': C_OPTS,
            'kernel': ['sigmoid'],
            'gamma': GAMMA_OPTS,
            'coef0': COEF0_OPTS,
            'shrinking': [True, False],
            'class_weight': [None, 'balanced'],
            'break_ties': [True, False],
        }, {
            'C': C_OPTS,
            'kernel': ['poly'],
            'degree': [2, 3, 4],
            'gamma': GAMMA_OPTS,
            'coef0': COEF0_OPTS,
            'shrinking': [True, False],
            'class_weight': [None, 'balanced'],
            'break_ties': [True, False],
        },
    ],
}
