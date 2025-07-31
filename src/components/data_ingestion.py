import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass
class DATAINGESTATIONCONFIG:
    raw_data_path : str= os.path.join('artifacts','Raw.csv')
    test_data_path :str= os.path.join('artifacts','test.csv')
    train_data_path: str= os.path.join('artifacts','train.csv')


class DATAINGESTION:
    def __init__(self):
        self.data_ingestion_config = DATAINGESTATIONCONFIG()

    def initialize_data_ingestion(self):
        try:
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            logging.info('Created directory for raw data and other files too')

            df= pd.read_csv(self.data_ingestion_config.raw_data_path)
            logging.info('Read the data from raw csv file')

            train_dataset, test_dataset = train_test_split(df, random_state=42, test_size=0.2)

            train_dataset.to_csv(self.data_ingestion_config.train_data_path, index= False)
            test_dataset.to_csv(self.data_ingestion_config.test_data_path, index= False)
            logging.info('Train adn test datasets are saved successfully')

            return (
                self.data_ingestion_config.raw_data_path,
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e. sys)


