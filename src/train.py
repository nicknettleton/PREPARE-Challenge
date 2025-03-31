
from loguru import logger
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import typer

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, GroupKFold
from sklearn.metrics import root_mean_squared_error

from mapie.regression import MapieRegressor
import shap

import src.defaults as defaults
from src.prepare_data import prepare_data

xgb.set_config(verbosity=2)

#------------------------------------------------------------------------------

def train(
    features_path = defaults.train_features_path,
    labels_path = defaults.train_labels_path,
    model_save_path = defaults.model_save_path,
    cv = False,
    cv_predict = False,
    cv_predictions_path = None,
    debug = False
):
    """Trains a new model and saves it to model_save_path.

    Args:
        features_path: path to the train set features file
        labels_path: path to the train set labels file
        model_save_path: where to save the model pickle file
        cv: whether to cross validate the model before fitting. The metric is reported with logger.
        cv_predict: whether to generate cross validation predictions for analysis or debugging
        cv_predictions_path: where to save cv predictions if cv_predict is True
        debug: whether to run in debug model

    Returns:
        ensemble (sklearn.ensemble.VotingRegressor): fitted ensemble for predicting labels
        mapie_regressor (mapie.regression.MapieRegressor): fitted MAPIE regressor for predicting intervals
    """
    
    n_rows = None
    if debug:
        logger.info('Running in debug mode')
        n_rows = 20

    # Load & prepare data

    logger.info(f'Loading data from {features_path}, {labels_path}')
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    logger.info(f'Processing data. Data shape: features {features.shape}, labels {labels.shape}')
    features = features.sample(frac=1, random_state=defaults.SEED)
    X, y, uids = prepare_data(features, labels)

    if debug:
        logger.info(f'Using only {n_rows} samples in debug mode')
        X = X.head(n_rows)
        y = y.head(n_rows)
        uids = uids.head(n_rows)
        
    # Define model

    # LGBMR
    
    lgbmr = lgb.LGBMRegressor(
        random_state=defaults.SEED, 
        metric='RMSE',
        verbosity=-1,
        n_estimators = 3000,
        learning_rate = 0.09575168925821444,
        num_leaves = 2160,
        max_depth = 18,
        min_data_in_leaf = 320,
        lambda_l1 = 30,
        lambda_l2 = 65,
        min_gain_to_split = 7.675060186062048,
        bagging_fraction = 0.9000000000000001,
        bagging_freq = 1,
        feature_fraction = 0.8    
    )

    # CBR

    cbr = CatBoostRegressor(
        random_seed=defaults.SEED,
        loss_function='RMSE',
        verbose=False,
        cat_features = list(X.select_dtypes(include=['category'])),
        nan_mode='Min',
        early_stopping_rounds=100,
        iterations=3000,
        learning_rate=0.01182984692550002,
        depth=6,
        subsample=0.6200067992114794,
        colsample_bylevel=0.1984151022967874,
        min_data_in_leaf=6,
    )

    # XGBR

    xgbr = xgb.XGBRegressor(
        tree_method="hist",
        eval_metric=root_mean_squared_error,
        enable_categorical=True,
        random_state=defaults.SEED,
        verbosity=0,
        n_estimators=700,
        max_depth=4,
        learning_rate=0.014,
        colsample_bytree=0.4,
        subsample=0.7,
        reg_alpha=5.389473619539341,
        reg_lambda=0.0029906193719320124,
        gamma=3,
        min_child_weight=10
    )

    # Ensemble

    estimators = [
        ('cbr', cbr),
        ('lgbmr', lgbmr),
        ('xgbr', xgbr)
    ]
    ensemble = VotingRegressor(estimators=estimators)

    # Cross validate

    if cv is True:
        logger.info(f'Cross validating')
        cv_results = cross_validate(
            ensemble, 
            X, 
            y, 
            scoring=defaults.cv_scoring, 
            cv=GroupKFold(defaults.cv_k),
            groups=uids,
            # return_train_score=True
        )
        logger.info(f'CV RMSE mean {-cv_results['test_score'].mean()}, std {cv_results['test_score'].std()}')

    # Get CV predictions
    # Note we shuffled the train data so they will be in a different order to the original train labels
    
    if cv_predict is True:
        logger.info(f'Generating CV predictions')
        y_pred = cross_val_predict(
            ensemble, 
            X, 
            y=y,
            cv=GroupKFold(defaults.cv_k),
            groups=uids
        )
        y_pred = y_pred.round(0).astype(int)
        pred_labels = pd.DataFrame({
            'uid': uids,
            'year': X['year'],
            'composite_score': y_pred
        })

        # Save if path given
        if cv_predictions_path is not None:
            logger.info(f'Saving CV predictions to {cv_predictions_path}')
            pred_labels.to_csv(cv_predictions_path, index=False)

        else:
            logger.info(f'cv_predictions_path not given')

    # Train ensemble

    logger.info('Training ensemble')
    ensemble.fit(X, y)

    # Define and train MAPIE for prediction intervals

    logger.info('Training MAPIE regressor')
    mapie_regressor = MapieRegressor(
        estimator=ensemble,
        method='plus',
        cv=GroupKFold(n_splits=defaults.cv_k),
        random_state=defaults.SEED
    )
    mapie_regressor = mapie_regressor.fit(
        X, y,
        groups=uids,
        random_state=defaults.SEED
    )

    # SHAP is applied in predict() on the fitted ensemble.estimators_

    # Save the bits we need

    if not debug:
        to_save = {
            'ensemble': ensemble,
            'mapie_regressor': mapie_regressor
        }
        with open(model_save_path, "wb") as file:
            pickle.dump(to_save, file)
        logger.success(f'Trained models saved to {model_save_path}')
    else:
        logger.success(f'Done')

    return ensemble, mapie_regressor

#------------------------------------------------------------------------------

def main(
    features_path: Path = typer.Option(
        defaults.train_features_path,
        help="Path to the raw training dataset for processing",
    ),
    labels_path: Path = typer.Option(
        defaults.train_labels_path, help="Path to the training labels"
    ),
    model_save_path: Path = typer.Option(
        defaults.model_save_path, help="Path to save the trained model weights"
    ),
    cv: bool = typer.Option(
        False, help="Cross validate on training dataset and report RMSE before training"
    ),
    cv_predict: bool = typer.Option(
        False, help="Generate predictions from cross validation and save before training"
    ),
    cv_predictions_path: Path = typer.Option(
        defaults.cv_predictions_path, help="Path to save predictions from cross validation"
    ),
    debug: bool = typer.Option(
        False, help="Run on a small subset of the data for debugging"
    ),
):
    train(
        features_path, labels_path, model_save_path,
        cv=cv, cv_predict=cv_predict, cv_predictions_path=cv_predictions_path,
        debug=debug
    )

#------------------------------------------------------------------------------

if __name__ == "__main__":
    typer.run(main)