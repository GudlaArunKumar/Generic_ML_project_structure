import os 
import sys 
import pandas as pd 
from src.exception import CustomException
from src.logger import logging 
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass 

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    """
    Class to hold paths of data set
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # stores all the paths of data

    def initiate_data_ingestion(self):
        # To read the data 
        logging.info("Data Ingestion process initiated ...")
        try:
            df = pd.read_csv('notebook\data\stud.csv') 
            logging.info("Reading the csv data as a dataframe...")

            # creating a folder to save the loaded df as csv file
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # saving train and test data into artifacts folder
            logging.info("Train test split initiated..")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion process is completed...")

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # rasing an exception using defined customexception class
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr, _))




