import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterGrid
from utils.metrics import accuracy_score 
from utils.helper import standard_scale
from tqdm import tqdm


def nested_cross_validation(X, y, model, param_grid, outer_k=5, inner_k=5, 
                          scoring=None, random_state=None, clone_func=None):
    """
    Generic nested cross-validation for any model
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    model : estimator object
        Model to evaluate (must implement fit and predict)
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings
    outer_k : int, default=5
        Number of outer CV folds
    inner_k : int, default=5
        Number of inner CV folds
    scoring : callable, default=None
        Scoring function (if None, uses model.score if available, else accuracy)
    random_state : int, default=None
        Random seed for reproducibility
    clone_func : callable, default=None
        Custom clone function if model doesn't support sklearn's clone
        
    Returns:
    --------
    outer_scores : list
        Scores for each outer fold
    best_params : list
        Best parameters for each outer fold
    """
    
    # Default scoring function
    if scoring is None:
        if hasattr(model, 'score'):
            scoring = lambda model, X, y: model.score(X, y)
        else:
            scoring = lambda model, X, y: np.mean(model.predict(X) == y)
    
    # Default clone function
    if clone_func is None:
        try:
            clone(model)
            clone_func = clone
        except TypeError:
            def clone_func(estimator):
                new_estimator = type(estimator)(**estimator.get_params())
                if hasattr(estimator, 'classes_'):
                    new_estimator.classes_ = estimator.classes_
                return new_estimator
    
    outer_scores = []
    best_params = []

    # Convert to arrays for consistent indexing
    X = np.array(X)
    y = np.array(y)

    outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)

    for train_idx, test_idx in tqdm(outer_cv.split(X), desc="Outer CV", total=outer_k):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        best_inner_score = -np.inf
        best_inner_params = None

        param_combinations = list(ParameterGrid(param_grid))

        for params in tqdm(param_combinations, desc="Inner Grid Search", leave=False):
            inner_scores = []
            inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=random_state)
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
                current_model = clone_func(model)
                current_model.set_params(**params)
                
                current_model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                score = scoring(current_model, X_train[inner_val_idx], y_train[inner_val_idx])
                inner_scores.append(score)
            
            mean_score = np.mean(inner_scores)
            
            if mean_score > best_inner_score:
                best_inner_score = mean_score
                best_inner_params = params
        
        best_model = clone_func(model)
        best_model.set_params(**best_inner_params)
        best_model.fit(X_train, y_train)
        
        outer_score = scoring(best_model, X_test, y_test)
        
        outer_scores.append(outer_score)
        best_params.append(best_inner_params)
    
    return outer_scores, best_params


def generate_parameters(param_grid):
    """
    Generate parameter combinations from grid (simple version)
    For full grid search, use sklearn's ParameterGrid instead
    """
    from sklearn.model_selection import ParameterGrid
    return list(ParameterGrid(param_grid))
