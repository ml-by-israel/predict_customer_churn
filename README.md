# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The objective of this project is to develop a production-ready codebase using industry-standard best practices. Specifically, the project focuses on predicting customer churn for banking clients, a classification task that involves identifying customers who are likely to discontinue their business with the bank.

## Files and data description
Overview of the files and data present in the root directory. 
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── Churn.png
│   │   ├── Customer_Age.png
│   │   ├── Gender.png
│   │   ├── heatmap.png
│   │   ├── Marital_Status.png
│   │   └── Total_Trans_Amt.png
│   └── results
│       ├── feature_importances.png
│       ├── logistic_results.png
│       ├── rf_results.png
│       └── roc_curve_result.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── __pycache__
│   ├── churn_library2.cpython-36.pyc
│   └── churn_library.cpython-36.pyc
├── README.md
└── requirements.txt

The repository contains 4 main folders:

- data: This folder contains the data saved in csv format.
-images: This folder contains two sub-folders, eda and results. The eda sub-folder stores visualizations of numerical, categorical, and heatmap data distributions of correlation values between variables. The results sub-folder contains visualizations and evaluations of models, such as ROC-curve, feature importance, and confusion matrix cards.
- logs: This folder stores the logs of function test results on the churn_library.py file.
- models: This folder is used to store model objects with the .pkl extension.
Other important files in the repository include:

- churn_library.py: This file contains the functions for reading data, exploratory data analysis, feature engineering (one-hot-encoding, standardization, and data splitting), and modeling with logistic regression and random forest (training, prediction, and evaluation).
- churn_script_logging_and_tests.py: This file is used for testing and logging the workflow modeling written in churn_library.py.
- requirements.txt: This file contains the libraries used in modeling and their versions.
- churn_notebook.ipynb: This file contains the prototype of the modeling workflow before it was converted into a function in churn_library.py.

## Running Files
How do you run your files? What should happen when you run your files?

To be able to run this project, you must install python library using the following command:

pip install -r requirements.txt

To run the workflow open the terminal in the root folder and use the command below:

ipython churn_library.py


 

