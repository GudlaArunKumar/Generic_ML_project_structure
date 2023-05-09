import os 
import sys 
import pandas as pd 
from src.exception import CustomException
from src.logger import logging  

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer 

class TrainPipeline: 
    """
    Class to initiate training process and store the pre processor and trained best model
    as a pickle files inside artifacts folder for prediction purpose.
    """

    def __init__(self):
        pass 

    def initiate_training(self):

        logging.info("Training pipeline initiated..")

        try:
            # data ingestion process
            obj = DataIngestion()
            train_path, test_path = obj.initiate_data_ingestion()

            # Data pre processing stage
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

            # model training stage
            modeltrainer = ModelTrainer()
            print(modeltrainer.initiate_model_trainer(train_arr, test_arr, _))

            logging.info("Training pipeline process completed..")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_obj = TrainPipeline()
    train_obj.initiate_training()