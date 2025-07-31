from src.logger import logging 
from src.exception import CustomException

import sys
import os
# from data ingestion
from src.components.data_ingestion import DATAINGESTION
# from data transformation
from src.components.data_transformation import DATA_TRANSFORMATION
# from data trainer
from src.components.model_trainer import MODEL_TRAINER



# Just for testing the code
# This part is commented out to avoid execution errors when running the script directly.
# Uncomment the following lines to test the code directly.
'''if __name__ == '__main__':
    try:
        logging.info('Starting the application')
        a = 1 / 0
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
    logging.info('Application finished with errors')'''

if __name__ == '__main__':
    try:
        logging.info("Staring data ingestion")
        data_ingestion_obj = DATAINGESTION()
        data_ingestion_obj.initialize_data_ingestion()

    except Exception as e:
        logging.error(f'An error occured during data ingestion {e}')
        raise CustomException (e,sys)
    

    try:
        logging.info('Starting data transformation stage')
        data_transformation_obj = DATA_TRANSFORMATION()
        train_arr, test_arr, preprocessing_obj = data_transformation_obj.preprocessing_initializer_function(train_data=data_ingestion_obj.data_ingestion_config.train_data_path,
                                                                                                                test_data=data_ingestion_obj.data_ingestion_config.test_data_path)
        logging.info('Data transformation completed Successufully')
    except Exception as e:
        raise CustomException(e,sys)
    

    try:
        logging.info('Model Trainer Stage')
        model_trainer = MODEL_TRAINER()
        score = model_trainer.model_trainer_initiator(train_arr, test_arr)
        print(score)
        logging.info(f'The model evalution stage has been executed successfully and the score is :{score}%')

    except Exception as e:
        raise CustomException(e,sys)
