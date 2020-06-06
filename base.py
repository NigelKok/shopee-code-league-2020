from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge
from sklearn.model_selection import train_test_split 
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import os

# Setup loggers
logger = logging.getLogger()
logger.handlers = [] 
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s: %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class ModelBase():
    
    def setModel(self):
        return Ridge()
    
    def setParamGrid(self):
        return {'reg__alpha':[0.001,0.1,10,100,10e5]}
    
    def setSteps(self, model):
        return [('scaler', StandardScaler()), ('reg', model)]
    
    def runModel(self):
        global X_test, y_test
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(dir_path)
        logging.info('Current path: {}'.format(os.getcwd()))
        
        # Prepare the data
        inputData, labels = self.getCleanData()
        X_train, X_test, y_train, y_test = train_test_split(inputData, labels,test_size=0.2, random_state=30)
        
        # Set model and assemble pipeline
        model = self.setModel() or Ridge()
        logging.info('Model: {}'.format(model))
        steps = self.setSteps(model)
        pipeline = Pipeline(steps)
        
        # Define and execute GridSearchCV
        parameters = self.setParamGrid()
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        grid.fit(X_train, y_train)
        
        res = grid.predict(X_test)
        logging.info('=========== Model Results ===========')
        logging.info('Predicted values: {}'.format(res))
        logging.info('Actual values: {}'.format(y_test))
        logging.info('Score = {}'.format(grid.score(X_test,y_test)))
        logging.info('Best params: {}'.format(grid.best_params_))
        return grid
    
    def getCleanData(self):
        dataset = pd.read_csv('salary_data.csv')
        inputData = dataset.iloc[:, :-1].values
        labels = dataset.iloc[:, 1].values
        return inputData, labels
        
if __name__=='__main__':
    testModel = ModelBase()
    trainedModel = testModel.runModel()
    