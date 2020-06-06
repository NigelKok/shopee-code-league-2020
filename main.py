from base import *

'''
The CustomModel should be used to override the methods in ModelBase

getCleanData: Reads the data from .xlsx/.csv file and returns a dataframe/numpy
array in the form of inputs (x) and labels (y)

setModel: Basic sklearn model to use, no model arguments needed

setParamGrid: The grid for GridSearchCV to iterate over, the logs should print 
out the attributes that can be tweaked

setSteps: Set the steps to be used in the pipeline, default is just a scaler +
model
'''

class CustomModel(ModelBase):
    def __init__(self):
        pass



def main():
    model = CustomModel()
    trainedModel = model.runModel()

main()