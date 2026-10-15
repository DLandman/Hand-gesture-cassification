import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.callback import ProgressBar
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score, GridSearchCV, GroupKFold, RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline, make_union, Pipeline
from sklearn.svm import SVC

import config
from preprocess import get_data



def get_param_grid(pipeline):
    param_grid = {}
    names = get_step_names(pipeline)
    for i, key in enumerate(names):
        if i == len(names) - 1:  # If it is the last step
            prefix = f"model__" if config.CUSTOM_NAMES else f"{key}__"
        elif len(names) > 2:  # If there are multiple transformers
            prefix = f"features__transformer_{i}__" if config.CUSTOM_NAMES else f"featureunion__{key}__"
        else:  # If there is only one transformer
            prefix = f"transformer__" if config.CUSTOM_NAMES else f"{key}__"
        try:
            for param_name, param_values in config.OPTIONS[key].items():
                param_grid[prefix + param_name] = param_values
        except KeyError:
            print(f"Warning: No tunable parameters found for step '{key}' in OPTIONS dictionary.")
    return param_grid

def get_pipeline(*transformers, model):
    assert len(transformers) == len(set(str(i).split('(')[0] for i in transformers)), "Feature sets must be disjoint. Please ensure that the transformers provided are unique."
    if transformers:
        if len(transformers) == 1:
            return Pipeline([('transformer', *transformers), ('model', model)]) if config.CUSTOM_NAMES else make_pipeline(*transformers, model)
        return Pipeline([('features', FeatureUnion([(f'transformer_{i}', tf) for i, tf in enumerate(transformers)])), ('model', model)]) if config.CUSTOM_NAMES else make_pipeline(make_union(*transformers), model)
    return Pipeline([('model', model)]) if config.CUSTOM_NAMES else make_pipeline(model)

def get_step_names(pipeline):
    step_names = []
    for _, step in pipeline.named_steps.items():
        if isinstance(step, FeatureUnion):
            for _, sub_step in step.transformer_list:
                step_names.append(sub_step.__class__.__name__.lower())
        else:
            step_names.append(step.__class__.__name__.lower())
    return step_names

def tune_pipeline_parameters(pipeline, X, y, groups, param_grid=None, n_trials=config.N_TRIALS, n_splits=config.N_SPLITS, n_jobs=None, verbosity:int|str='pbar'):
    best_model = {'best_estimator_': None, 'best_params_': None, 'best_score_': -np.inf}
    param_grid = get_param_grid(pipeline) if param_grid is None else param_grid
    for trial in range(n_trials):
        grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42*trial), n_jobs=n_jobs)
        grid_search.set_params(**{'verbose': verbosity}) if not verbosity == 'pbar' else grid_search.set_callbacks(ProgressBar(max_propagation_depth=0))
        grid_search.fit(X, y, groups=groups)
        if grid_search.best_score_ > best_model['best_score_']:
            best_model['best_estimator_'] = grid_search.best_estimator_
            best_model['best_params_'] = grid_search.best_params_
            best_model['best_score_'] = grid_search.best_score_
    return best_model



if __name__ == "__main__":
    X, y, groups, raw, cols = get_data()
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, test_size=0.2, random_state=42, stratify=y)

    pipeline = get_pipeline(PCA(), SelectKBest(), model=SVC())
    best_model = tune_pipeline_parameters(pipeline, X_train, y_train, groups_train, n_trials=1, n_jobs=-1)

    print(f"Best Parameters: {best_model['best_params_']}")
    print(f"Best Validation Score: {best_model['best_score_']:.4f}")
    print(f"Test Score: {best_model['best_estimator_'].score(X_test, y_test):.4f}")
