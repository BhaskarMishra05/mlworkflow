from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pickle


@dataclass
class DATA_TRANSFORMATION_CONFIG:
    processed_data_file_path = os.path.join('artifacts', 'preprocessed.pkl')

class DATA_TRANSFORMATION:
    def __init__(self):
        self.data_trasformation_config = DATA_TRANSFORMATION_CONFIG()

    def preprocessing_function(self, df: pd.DataFrame):

        try:
            logging.info('Staring preprocessing function')
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
            logging.info(f' Numerical Columns: {numerical_columns}')
            logging.info(f' Categorical Columns: {categorical_columns}')
            numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                                 ('scaler', StandardScaler())])
            categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                   ('encoder', OneHotEncoder(drop= 'first', handle_unknown='ignore'))])
    
            logging.info('Created numerical and categorical pipelines')

            preprocessing = ColumnTransformer(transformers=[('numerical_pipeline', numerical_pipeline, numerical_columns),
                                                            ('categorical_pipeline', categorical_pipeline, categorical_columns)])
            
            logging.info('Created preprocessing pipeline with numerical and categorical column transformers')

            logging.info('Sucessfully created the preprocessing pipeline')
            return preprocessing

        except Exception as e:
            raise CustomException(e,sys)
    
    def preprocessing_initializer_function(self, train_data, test_data):

        try:
            logging.info('Starting the initializer function for  preprocessing')
            train= pd.read_csv(train_data)
            test= pd.read_csv(test_data)
            logging.info('Successfully read and loaded the train and test data')

            TARGET='math score'
            logging.info(f'Target columns is {TARGET}')
            logging.info('Splitting the data into features and target variables from both train and test dataset')

            train_features = train.drop(columns=[TARGET], axis=1)
            train_target = train[TARGET]
            test_features = test.drop(columns=[TARGET], axis=1)
            test_target = test[TARGET]
            logging.info('Successfully split the data')
            preprocessing_obj = self.preprocessing_function(train_features)
            preprocessed_array_train = preprocessing_obj.fit_transform(train_features)
            preprocessed_array_test = preprocessing_obj.transform(test_features)

            logging.info('Successfully applied the preproocesssed techniques on train and test')

            train_array = np.c_[preprocessed_array_train, train_target]
            test_array = np.c_[preprocessed_array_test, test_target]

            logging.info('successfully concatenated the preprocessed features of train and test wtih thier respective target values')
            
            joblib.dump(preprocessing_obj, self.data_trasformation_config.processed_data_file_path)
            logging.info('Successfully saved the preprocessing.pkl file in the artifacts folder')

            return (train_array, test_array, preprocessing_obj)
        
        except Exception as e:
            raise CustomException(e,sys)

