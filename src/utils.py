import dill
import os 
import sys
import joblib
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DATAINGESTION
from src.components.data_transformation import DATA_TRANSFORMATION



from sklearn.metrics import (
    r2_score, root_mean_squared_error, mean_squared_error
)
from sklearn.model_selection import GridSearchCV


data_ingestion = DATAINGESTION()
data_transformation = DATA_TRANSFORMATION
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def model_evaluation(X_train, y_train, X_test, y_test, models , params):
    logging.info('Runnnig model evalution function from Utils')
    try:
        R2 = {}
        rmse_score = {}
        best_model_dict = {}
        for name, model in models.items():
            if params.get(name):
                gcv = GridSearchCV(model, param_grid=params[name], cv=5)
                gcv.fit(X_train, y_train)
                best_model = gcv.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            R2[name] = r2_score(y_test, y_pred)
            rmse_score[name] = root_mean_squared_error(y_test, y_pred)
            best_model_dict[name] = best_model
        logging.info('The function has been executed successfully')
        return dict(R2), dict(rmse_score) , dict(best_model_dict)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)