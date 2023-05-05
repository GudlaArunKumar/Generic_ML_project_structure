import os 
import sys
from dataclasses import dataclass 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import r2_score

from src.exception import CustomException 
from src.logger import logging 
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path): 
        try:
            logging.info("Splitting train and test array into features and target..")
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1], 
                                                test_arr[:,:-1], test_arr[:,-1])
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree":  DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ada boost": AdaBoostRegressor(),
                "Linear Regressor": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor() 
            }

            model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                          models=models)
            
            # to get best model's test R2 score
            best_r2_score = max(model_report.values())

            # to get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_r2_score)]  
            best_model = models[best_model_name]

            if best_r2_score < 0.60:
                raise CustomException("No best model found ")
            
            logging.info("Training model is completed and found the best model...")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model)   

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square   


        except Exception as e:
            raise CustomException(e, sys)
            








