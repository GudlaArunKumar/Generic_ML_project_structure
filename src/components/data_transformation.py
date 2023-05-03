import sys 
import os
from dataclasses import dataclass 
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer

from src.exception import CustomException 
from src.logger import logging 
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """ To store path of outputs in data transformation process """
    preprocessor_obj_file_path = os.path.join('artifacts', 'pre_processor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() 

    def pre_process_data_object(self):
        """
        This function returns pre processing object 
        """
        try:
            num_features = ['reading_score', 'writing_score'] 
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
       'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("cat_imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)) # optional to use
                ]
            )

            logging.info("Data Preprocessing defined....")

            preprocessor = ColumnTransformer([
                ("num_pipelines", num_pipeline, num_features),
                ("cat_pipelines", cat_pipeline, cat_features)
            ], remainder='passthrough')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
            

    def initiate_data_transformation(self, train_path, test_path):
        """
        Function to perform pre-processing on train data and returns the 
        pre processed object
        """

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data for data transformation completed...")

            logging.info("Obtaining preprocessing object...")

            preprocessor_obj = self.pre_process_data_object()

            target_column = "math_score"
            features_train_df = train_df.drop([target_column], axis=1)
            target_train_df = train_df[target_column]

            features_test_df = test_df.drop([target_column], axis=1)
            target_test_df =  test_df[target_column]

            logging.info("Applying pre processor on train and test data")

            input_feature_train_arr = preprocessor_obj.fit_transform(features_train_df)
            input_feature_test_arr = preprocessor_obj.transform(features_test_df)

            # Combining preprocessed data and target column as a array
            train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]

            logging.info("Saving preprocessor object as a pickle file  ")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
            




