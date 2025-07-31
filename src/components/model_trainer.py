import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import model_evaluation, save_object
from dataclasses import dataclass
from sklearn.metrics import r2_score
# Importing models to be used for training
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class MODEL_TRAINER:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def model_trainer_initiator(self, train_array, test_array):

        try:
            logging.info('Successfully initiated model trainer class')

            X_train, y_train, X_test, y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'RandomForestRegressor': RandomForestRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor()
            }

            params = {
                'LinearRegression': {
                    'fit_intercept': [True, False]
                },
                'RandomForestRegressor':{
                    'n_estimators': [5,10,15],
                    'max_depth': [None, 5,7,10],
                    'min_samples_leaf': [1,2,4]
                },
                'DecisionTreeRegressor': {
                    'splitter': ['best','random'],
                    'max_depth': [None,5,7,10],
                    'min_samples_leaf': [1,2,4]
                },
                'KNeighborsRegressor': {
                    'n_neighbors': [3,5,7,9],
                    'weights': ['uniform','distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p': [1,2]
                }
            }
            logging.info('Defined models and params')
            logging.info('Initating model_evalution function from utils')
            r2_report, rmse_report, trained_models= model_evaluation(X_train, y_train, X_test, y_test, models, params)
            logging.info('Sorting best scores to sort the best model out of all the mdoels')
            best_r2_score = max(sorted(r2_report.values()))
            best_rmse_score = min(sorted(rmse_report.values()))
            logging.info(f'Best score is {best_r2_score}')
            logging.info('Finding the best model name')
            best_model_name = list(r2_report.keys())[
                list(r2_report.values()).index(best_r2_score)
            ]
            best_model = trained_models[best_model_name]
            logging.info(f'Best model is {best_model}')

            if best_r2_score <0.6:
                raise CustomException('No best model found')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)


            logging.info('Successfully executed the Model Trainer pipeline')
            return r2
        except Exception as e:
            raise CustomException(e,sys)