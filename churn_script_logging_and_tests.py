'''
Unit tests for churn_library functions.

Author: Israel Osorio
Date: Mar 2023
'''

import os
import logging
import pandas as pd
import pytest
from churn_library import import_data, perform_eda, encoder_helper, 
perform_feature_engineering, train_models
from constants import CAT_COLUMNS

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.mark.usefixtures("cache")
def test_import(request):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(r"./data/bank_data.csv")
        request.config.cache.set('cache_df', df.to_dict())
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.mark.usefixtures("cache")
def test_eda(request):
    '''
    test perform eda function
    '''
    cached_dict = request.config.cache.get('cache_df', None)
    df = pd.DataFrame.from_dict(cached_dict)
    perform_eda(df)

    try:
        assert os.path.exists('./images/eda/total_trans_ct.png')
        assert os.path.exists('./images/eda/heat_map.png')
        assert os.path.exists('./images/eda/churn_hist.png')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: One or more of plots were not created")
        raise err

    # Delete the images after testing
    os.remove('./images/eda/churn_hist.png')
    os.remove('./images/eda/total_trans_ct.png')
    os.remove('./images/eda/heat_map.png')


@pytest.mark.usefixtures("cache")
def test_encoder_helper(request):
    '''
    test encoder helper
    '''

    cached_dict = request.config.cache.get('cache_df', None)
    df = pd.DataFrame.from_dict(cached_dict)
    df = encoder_helper(df, CAT_COLUMNS)
    request.config.cache.set('cache_encoded_df', df.to_dict())

    try:
        assert df.select_dtypes(
            include=[
                'category',
                'object']).columns.tolist() == []
        logging.info("Testing test_encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: Not all columns were encoded")
        raise err


@pytest.mark.usefixtures("cache")
def test_perform_feature_engineering(request):
    '''
    test perform_feature_engineering
    '''
    cached_dict = request.config.cache.get('cache_encoded_df', None)
    df = pd.DataFrame.from_dict(cached_dict)

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(df)
        request.config.cache.set('X_train', x_train.to_dict())
        request.config.cache.set('X_test', x_test.to_dict())
        request.config.cache.set('y_train', y_train.to_dict())
        request.config.cache.set('y_test', y_test.to_dict())
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing test_perform_feature_engineering: Required columns are not in index of df")
        raise err

@pytest.mark.usefixtures("cache")
def test_train_models(request):
    '''
    test train_models
    '''

    X_train = pd.DataFrame.from_dict(request.config.cache.get('X_train', None))
    X_test = pd.DataFrame.from_dict(request.config.cache.get('X_test', None))
    y_train = pd.Series(request.config.cache.get('y_train', None))
    y_test = pd.Series(request.config.cache.get('y_test', None))

    train_models(X_train, X_test, y_train, y_test)

    try:
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./models/logistic_model.pkl')
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Model were not saved in ./models")

        raise err

    try:
        assert os.path.exists('./images/results/classification_report_rf.png')
        assert os.path.exists('./images/results/classification_report_lr.png')
    except AssertionError as err:
        logging.error(
            "Testing train_models: Classification reports were not saved in ./images/results/")
        raise err

    try:
        assert os.path.exists('./images/results/feature_importance.png')
    except AssertionError as err:
        logging.error(
            "Testing train_models: Feature importance plot was not saved in ./images/results/")
        raise err

    os.remove('./models/rfc_model.pkl')
    os.remove('./models/logistic_model.pkl')
    os.remove('./images/results/feature_importance.png')
    os.remove('./images/results/classification_report_lr.png')
    os.remove('./images/results/classification_report_rf.png')


if __name__ == "__main__":
    pytest.main(
        args=["-v",
              "churn_script_logging_and_tests.py",
              "--cache-clear"])
