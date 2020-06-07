# shopee-code-league-2020

Problem types and related keywords (try to look up the keywords if you think a problem falls under that category):

Main libraries: sklearn, keras, pandas

1. Supervised learning - Data (x values) and labels (y values) are provided for training
  a. Classification - eg. Is this picture a cat or dog?
    - Image data: Keras flow_from_directory
    - Non-image data: One-hot encoding for labels
    - Word data: bag-of-words
    - Preprocessing: StandardScaler for all data types by default if unsure
    - Models: Logistic regression, convolutional neural networks, support vector machines, VGG16, Resnet
  b. Regression - Prediction of a continuous value eg. Price of house based on XX factors
    - Preprocessing: StandardScaler for all data types by default if unsure
    - Models: Linear regression, ridge, lasso, artificial neural networks (custom-built)

2. Unsupervised learning - Only data (x values) are provided
    - Models: clustering algorithms (eg. K-means)
    
Basic Intro to Machine Learning
You have a set of data X and a set of labels Y and you want to build a model to find a relationship between X and Y so in future, when given random values of X, you can accurately predict what the Y is.

A model has specific parameters which tweaks itself during the model training process (read: gradient descent, cost function) but these happen in the backend. What we need to do is to find the optimal model (step 1) and the optimal set of hyperparameters for the chosen model (step 2). Sadly it's mostly trial and error.

The base class uses a Pipeline object to integrate the whole process and GridSearchCV to iterate on parameters (which have to be put in manually) and return the model with the best performing set of parameters. Main steps as follow:
1. Data extraction - You probably have to override getCleanData() for every case since data is presumably different each time
2. Model selection - Read on the models appropriate for the scenario and use setModel() to return a default instance of the model
3. Parameter selection - Iterate some model hyperparameters, will have to manually input via setParamGrid()

runModel() should return a trained model with the best set of parameters (based on what you input in 3. above). You can write code to continue using it for making predictions (you need to write some code to extract the X values to make predictions on).
