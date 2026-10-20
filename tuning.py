import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.callback import ProgressBar
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, GridSearchCV, GroupKFold, HalvingGridSearchCV, RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline, make_union, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import config
from preprocess import get_data



def get_param_grid(pipeline, warn_missing=True):
    param_grid = {}
    names = get_step_names(pipeline, prefix=True)
    for i, name in enumerate(names):
        prefix, key = name.rsplit('__', 1)
        try:
            if isinstance(param_grid, list):
                if isinstance(config.OPTIONS[key], list):
                    param_grid = [grid.copy() for grid in param_grid for _ in range(len(config.OPTIONS[key]))]
                    for k in range(0, len(param_grid), len(config.OPTIONS[key])):
                        for j, entry in enumerate(config.OPTIONS[key]):
                            param_grid[k + j].update({f"{prefix}__{param_name}": param_values for param_name, param_values in entry.items()})
                else:
                    for grid in param_grid:
                        grid.update({f"{prefix}__{param_name}": param_values for param_name, param_values in config.OPTIONS[key].items()})
            else:
                if isinstance(config.OPTIONS[key], list):
                    param_grid = [param_grid.copy() for _ in range(len(config.OPTIONS[key]))]
                    for j, entry in enumerate(config.OPTIONS[key]):
                        param_grid[j].update({f"{prefix}__{param_name}": param_values for param_name, param_values in entry.items()})
                else:
                    param_grid.update({f"{prefix}__{param_name}": param_values for param_name, param_values in config.OPTIONS[key].items()})
        except KeyError:
            if warn_missing:
                print(f"Warning: No tunable parameters found for step '{key}' in OPTIONS dictionary.")
    return param_grid

def get_pipeline(*transformers, model):
    steps = {}
    if transformers:
        for i, transformer in enumerate(transformers, start=1):
            if isinstance(transformer, (tuple, list)):
                # assert len(transformer) == len(set(str(j).split('(')[0] for j in transformer)), "Feature sets must be disjoint. Please ensure that the transformers provided are unique."
                steps[f"transformer_{i}"] = FeatureUnion([(f'transformer_{i}.{j}', tf) for j, tf in enumerate(transformer, start=1)]) if config.CUSTOM_NAMES else make_union(*transformer)
            else:
                steps[f"transformer_{i}"] = transformer
    if isinstance(model, (tuple, list)):
        # assert len(model) == len(set(str(j).split('(')[0] for j in model)), "Feature sets must be disjoint. Please ensure that the transformers provided are unique."
        if config.CUSTOM_NAMES:
            steps['ensemble_model'] = VotingClassifier([(f'model_{j}', m) for j, m in enumerate(model, start=1)])
        else:
            steps['ensemble_model'] = VotingClassifier([(m.__class__.__name__.lower(), m) for j, m in enumerate(model, start=1)])
    else:
        steps['model'] = model
    pipeline = Pipeline(list(steps.items())) if config.CUSTOM_NAMES else make_pipeline(*steps.values())
    assert len(get_step_names(pipeline)) == len(get_step_names(pipeline, prefix=True)), "Feature sets must be disjoint. Please ensure that the transformers provided are unique."
    return pipeline

def get_step_names(pipeline, prefix=False):
    step_names = {}
    for name, step in pipeline.named_steps.items():
        if isinstance(step, FeatureUnion):
            for sub_name, sub_step in step.named_transformers.items():
                step_names[sub_step.__class__.__name__.lower() if not prefix else f"{name}__{sub_name}__{sub_step.__class__.__name__.lower()}"] = sub_step
        elif isinstance(step, VotingClassifier):
            step_names[step.__class__.__name__.lower() if not prefix else f"{name}__{step.__class__.__name__.lower()}"] = step
            for sub_name, sub_step in step.named_estimators.items():
                step_names[sub_step.__class__.__name__.lower() if not prefix else f"{name}__{sub_name}__{sub_step.__class__.__name__.lower()}"] = sub_step
        else:
            step_names[step.__class__.__name__.lower() if not prefix else f"{name}__{step.__class__.__name__.lower()}"] = step
    return step_names

def tune_pipeline_parameters(pipeline, X, y, groups, search_strategy='grid', param_grid=None, factor=3, scoring=config.SCORING, refit=None, n_trials=1, n_splits=5, max_iter=1e5, n_jobs=None, verbosity:int|str='pbar'):
    best_model = {'best_estimator_': None, 'best_params_': None, 'best_score_': -np.inf}
    if not refit:
        refit = scoring[0] if not isinstance(scoring, (str, type(None))) else True
    param_grid = get_param_grid(pipeline) if param_grid is None else param_grid
    # pipeline.set_params(**{f'{list(param_grid.keys() if isinstance(param_grid, dict) else param_grid[-1].keys())[-1].rsplit("__", 1)[0]}__max_iter': int(max_iter)})
    for trial in range(n_trials):
        if search_strategy == 'grid':
            grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42*trial))
        elif search_strategy == 'halving':
            grid_search = HalvingGridSearchCV(pipeline, param_grid, factor=factor, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42*trial))
        else:
            raise ValueError("Invalid search strategy. Please choose 'grid' or 'halving'.")
        try:
            grid_search.set_params(**{'verbose': verbosity}) if not verbosity == 'pbar' else grid_search.set_callbacks(ProgressBar(max_propagation_depth=0))
            grid_search.fit(X, y, groups=groups)
            if grid_search.best_score_ > best_model['best_score_']:
                best_model['best_estimator_'] = grid_search.best_estimator_
                best_model['best_params_'] = grid_search.best_params_
                best_model['best_score_'] = grid_search.best_score_
        except KeyboardInterrupt:
            print(f"Interrupted during trial {trial}. Returning best model found so far.")
    return best_model



if __name__ == "__main__":
    X, y, groups, raw, cols = get_data()
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, test_size=0.2, random_state=42, stratify=y)

    pipeline = get_pipeline(MinMaxScaler(), (PCA(), SelectKBest()), model=SVC())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        best_model = tune_pipeline_parameters(pipeline, X_train, y_train, groups_train, search_strategy='halving', n_trials=1, n_jobs=-1)


    ConfusionMatrixDisplay.from_estimator(best_model['best_estimator_'], X_test, y_test)
    plt.show()

    print(f"Best Parameters: {best_model['best_params_']}")
    print(f"Best Validation Score: {best_model['best_score_']:.4f}")
    print(f"Test Score: {best_model['best_estimator_'].score(X_test, y_test):.4f}")
