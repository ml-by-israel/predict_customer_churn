# library doc string


# import libraries

# create this file to define constants
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
from constants import CAT_COLUMNS, QUANT_COLUMNS, KEEP_COLS, RANDOM_STATE
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df = df.drop('Attrition_Flag', axis=1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''


    # categorical plot
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Churn'])
    plt.savefig('./images/eda/churn_hist.png')
 
    # quant plot
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_trans_ct.png')
    
    # heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/heat_map.png')

def encoder_helper(df, category_lst, response='_Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for col in category_lst:
        col_lst = []
        col_groups = df.groupby(col).mean()['Churn']

        for val in df[col]:
            col_lst.append(col_groups.loc[val])
        df[col + response] = col_lst

    return df


def perform_feature_engineering(df, response=""):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    X = df[KEEP_COLS].copy()
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    
    plt.savefig(output_pth + '/feature_importance.png')
    

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    
    # models 
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000, random_state=RANDOM_STATE)
    
    # grid search
    param_grid = { 
    'n_estimators': [200, 500],
    'max_depth' : [4,5]
}

    
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    # plots
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)    
    plt.savefig('./images/results/roc_curves.png')
    
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    # feature importance plots 
    
    feature_importance_plot(cv_rfc.best_estimator_, X_train, './images/results')
if __name__ == '__main__':
    print("Reading data...")
    df = import_data(r"./data/bank_data.csv")
    print("Encoding...")
    df = encoder_helper(df, CAT_COLUMNS)
    print("Performing feature engineering")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    print("training models")
    train_models(X_train, X_test, y_train, y_test)
    # perform_eda(df)

    
