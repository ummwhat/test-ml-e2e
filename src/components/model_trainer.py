import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import( AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and testing input data")

            xtrain,ytrain,xtest,ytest = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            model_report:dict = evaluate_model(xtrain = xtrain,ytrain= ytrain, xtest = xtest,ytest =ytest,models = models)

            logging.info("evaluation reports created successfully")

            best_model_score = max(sorted(list(model_report.values())))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]


            best_model = models[best_model_name]

            if best_model_score<0.7:
                raise CustomException("No model performs well")
            
            logging.info("Best model has been found")

            
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            ypred = best_model.predict(xtest)

            return r2_score(ytest,ypred)

        
        
        

        except Exception as e:
            raise CustomException(e,sys)