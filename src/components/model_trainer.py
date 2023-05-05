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
                "Decision Tree":  DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "Ada boost": AdaBoostRegressor(),  
                "KNN Regressor": KNeighborsRegressor() 
            }

            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regressor":{},

                "Ada boost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "KNN Regressor": {'n_neighbors': [3,5,7,9,11]}
            }

            model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                          models=models, params=params)
            
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
            








