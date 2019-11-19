
# Python version
import sys
print('Python: {}'.format(sys.version))
import scipy
print('scipy: %s' % scipy.__version__)
import numpy as np
print('numpy: %s' % np.__version__)
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
import pandas as pd
print('pandas: %s' % pd.__version__)
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
import sklearn  #scikit-learn
print('sklearn: %s' % sklearn.__version__)
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
import keras
print('keras: %s' % keras.__version__)


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import svm
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)

# goal practice spot checking different machine learning algorithms
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

# Spot check linear algorithms on a dataset
# (e.g. linear regression, logistic regression and linear discriminate analysis)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
scoring = 'neg_mean_squared_error'  # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')  # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
scoring = 'neg_mean_squared_error'  # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()  # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
scoring = 'neg_mean_squared_error'  # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())



url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)

# Spot check some non-linear algorithms on a dataset (e.g. KNN, SVM and CART).


model = KNeighborsRegressor()
scoring = 'max_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

# model = svm.SVC()
# scoring = 'adjusted_mutual_info_score'  # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print(results.mean())

# Spot-check some sophisticated ensemble algorithms on a dataset
# (e.g. random forest and stochastic gradient boosting).
model = tree.DecisionTreeRegressor()  # https://scikit-learn.org/stable/modules/tree.html
scoring = 'r2'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())


well








