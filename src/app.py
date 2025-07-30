from src.logger import logging 
from src.exception import CustomException
import sys
import os
# from data ingestion
from src.components.data_ingestion import DATAINGESTION

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