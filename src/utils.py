import os 
import sys 

import numpy as np 
import pandas as pd 
import dill 
from sklearn.metrics import r2_score
from src.exception import CustomException 
from src.logger import logging
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    # function to store th pickle file in a given path

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param_grid = params[list(params.keys())[i]]

            # hyper parameter tuning using cross validation setting
            gs = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error')
            gs.fit(X_train, y_train)

            # using best tuned parameters from grid search CV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2_score = r2_score(y_train_pred, y_train)
            test_r2_score = r2_score(y_test_pred, y_test)

            report[list(models.keys())[i]] = test_r2_score

        logging.info("Hyper parameter tuning of models are completded..")

        return report 
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_obj(file_path):
    """ Function to load the pickle file and returns it"""

    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)   
    except Exception as e:
        raise CustomException(e, sys)
    



