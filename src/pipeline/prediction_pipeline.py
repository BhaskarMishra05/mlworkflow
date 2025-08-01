import os 
import sys
import pandas as pd
import numpy as np
from  src.exception  import CustomException
from src.logger import logging
from src.utils import load_object

logging.info('Starting Prediction Pipeline stage')
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        logging.info('Starting Preprocessing function')
        try:
            model_path = 'artifacts/model.pkl'
            preprocessed_path = 'artifacts/preprocessed.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessed_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            logging.info('Successfully ended preprocessing function')
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)
class CUSTOMDATA:
    def __init__(self,
                gender : str,
                race_ethnicity : str,
                parental_level_of_education : str,
                lunch : str,
                test_preparation_course : str,
                reading_score : int,
                writing_score : int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score 

    def get_data_as_data_frame(self):
        logging.info('Starting Dataframe making function')
        try:
            custon_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            logging.info('Successfully ended the dataframe making stage')
            return pd.DataFrame(custon_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)