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

from mapie.regression import MapieRegressor
import shap

import src.defaults as defaults
from src.prepare_data import prepare_data

xgb.set_config(verbosity=2)

#------------------------------------------------------------------------------

def predict(
    model_path = defaults.model_save_path,
    features_path = defaults.test_features_path,
    submission_save_path = defaults.test_predictions_path,
    submission_format_path = defaults.submission_format_path,
    include_mapie = False,
    debug = False
):
    """Makes predictions using the model at model_save_path.

    Returns the predictions, optionally with MAPIE intervals, and SHAP explanations.

    Args:
        model_path: path to the model pickle file
        features_path: path to the test/prediction set features file
        submission_save_path: where to save the submission to
        submission_format_path: path to the submission format file
        include_mapie: whether to include MAPIE intervals in the submission file and returned labels
        debug: whether to run in debug model

    Returns:
        pred_labels (pd.DataFrame): the predicted labels, and MAPIE intervals if requested
        ensemble_explanation (shap.Explanation): SHAP Explanation object for the overall ensemble
        subestimator_explanations (Dict): a dictionary of SHAP Explanation objects for each subestimator in the ensemble, with the subestimator names as the keys
    """

    n_rows = None
    if debug:
        logger.info('Running in debug mode')
        n_rows = 20
        
    # Load & prepare data

    logger.info(f'Loading data from {features_path}, {submission_format_path}')
    features = pd.read_csv(features_path)
    labels = pd.read_csv(submission_format_path)

    logger.info(f'Processing data. Data shape: features {features.shape}, labels {labels.shape}')
    X, y, uids = prepare_data(features, labels)

    if debug is True:
        logger.info(f'Using only {n_rows} samples in debug mode')
        X = X.head(20)
        y = y.head(20)
        uids = uids.head(20)

    # Load model

    logger.info(f'Loading trained model weights from {model_path}')
    with open(model_path, "rb") as file:
        saved = pickle.load(file)

    ensemble = saved['ensemble']
    mapie_regressor = saved['mapie_regressor']
        
    # Predict

    logger.info(f'Predicting labels')
    y_pred = ensemble.predict(X)
    y_pred = y_pred.round(0).astype(int)
    pred_labels = pd.DataFrame({
        'uid': uids,
        'year': X['year'],
        'composite_score': y_pred
    })

    # MAPIE prediction intervals
    # We do not set a min of 0 here, this will to be done prior to outputting

    if include_mapie:

        logger.info(f'Predicting MAPIE intervals')
        mapie_y_pred, mapie_intervals = mapie_regressor.predict(X, alpha=[0.05, 0.2])
    
        # pred_labels['mapie_y'] = mapie_pred_y.round(0).astype(int)
        pred_labels['mapie_lower_90'] = mapie_intervals[:,0,0].round(0).astype(int)
        pred_labels['mapie_upper_90'] = mapie_intervals[:,1,0].round(0).astype(int)
        pred_labels['mapie_lower_60'] = mapie_intervals[:,0,1].round(0).astype(int)
        pred_labels['mapie_upper_60'] = mapie_intervals[:,1,1].round(0).astype(int)

    # SHAP

    logger.info(f'Calculating SHAP values')
    ensemble_explanation, subestimator_explanations = shap_explanations(ensemble, X)
    
    # Save

    if not debug:
        pred_labels.to_csv(submission_save_path, index=False)
        logger.success(f'Submission saved to {submission_save_path}')
    else:
        logger.success(f'Done')

    return pred_labels, ensemble_explanation, subestimator_explanations

#------------------------------------------------------------------------------

def shap_explanations(ensemble, X):
    """Builds and returns the SHAP explanations for the predictions.

    Called by predict()

    Args:
        ensemble (sklearn.ensemble.VotingRegressor): fitted ensemble used to make predictions
        X (pd.DataFrame): prepared X for prediction
    
    Returns:
        ensemble_explanation (shap.Explanation): SHAP Explanation object for the overall ensemble
        subestimator_explanations (Dict): a dictionary of SHAP Explanation objects for each subestimator in the ensemble, with the subestimator names as the keys
    """

    # shap doesn't support VotingRegressor
    # so we build explainers for each estimator
    # then make the ensemble calculations by hand

    sub_explanations = {}
    sub_expected_values = []
    for name, estimator in ensemble.named_estimators_.items():
    
        explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
        explanation = explainer(X)
        sub_explanations[name] = explanation
        sub_expected_values.append(explainer.expected_value)
    
    # ensemble calculations
    # see:
    # https://arxiv.org/pdf/2106.08990
    # https://github.com/shap/shap/issues/112
    
    all_shap_values = [exp.values for exp in sub_explanations.values()] # the shap values from each estimator 
    ensemble_shap_values = np.mean(all_shap_values, axis=0) # an array of same shape as X with shap value for each sample and feature
    ensemble_expected_value = np.mean(sub_expected_values) # mean of mean predictions for each estimator

    # put the data in a shap.Explanation object
    # see:
    # https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html#shap.Explanation
    # https://github.com/shap/shap/blob/master/shap/_explanation.py
    
    ensemble_explanation = shap.Explanation(
        values=ensemble_shap_values,
        feature_names=list(X.columns),
        data=X.to_numpy(),
        base_values=ensemble_expected_value # np.full(len(shap_feature_names), ensemble_expected_value).squeeze()
        # Due to a possible bug in the shap waterfall code
        # explanation base_values needs to be a single numeric, not a list
        # this doesn't seem to affect other functionality; but if it does, revert to a list
        # and fix for the waterfalls
        # see https://github.com/shap/shap/issues/1801
    )

    return ensemble_explanation, sub_explanations

#------------------------------------------------------------------------------

def main(
    model_path: Path = typer.Option(
        defaults.model_save_path, help="Path to the saved model weights"
    ),
    features_path: Path = typer.Option(
        defaults.test_features_path, help="Path to the test features"
    ),
    submission_save_path: Path = typer.Option(
        defaults.test_predictions_path, help="Path to save the generated submission"
    ),
    submission_format_path: Path = typer.Option(
        defaults.submission_format_path, help="Path to the submission format csv"
    ),
    include_mapie: bool = typer.Option(
        False, help="Include MAPIE intervals in the submission"
    ),
    debug: bool = typer.Option(
        False, help="Run on a small subset of the data for debugging"
    ),
):
    predict(
        model_path, features_path, submission_save_path, submission_format_path,
        include_mapie=include_mapie,
        debug=debug
    )

#------------------------------------------------------------------------------

if __name__ == "__main__":
    typer.run(main)
