

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

from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Lesson 11: Improve Accuracy with Algorithm Tuning
# from lesson10, LDA and LR best algos
# tune algo parameters to specific dataset
# w scikit-learn, 2 ways:

# Tune the parameters of an algorithm using a grid search that you specify
from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# prepare models
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))  # creates a list of names with algos
models.append(('LDA', LinearDiscriminantAnalysis()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:  # for the name and the model in the list models...
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)  # results get stored here in a list, append adds new stuff to the end of the list
    names.append(name)  # adds the name to list names
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# # boxplot algorithm comparison
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(names)
# pyplot.show()

alphas = numpy.array([15, 10, 8, 5, 2, 1, 0.1])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, Y)
msggrid = "%s: %f, %f" % ('Ridge', grid.best_score_, grid.best_estimator_.alpha)
print(msggrid)

model = LogisticRegression(solver='liblinear')
param_grid = {'penalty': ['l1', 'l2'], 'C': [8, 9, 10]}
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
grid.fit(X, Y)
msggrid = "%s: best score: %f, best estimator: %f" % ('LogReg', grid.best_score_, grid.best_estimator_.C)
print(msggrid)

# LDA needs no tuning...

# Tune the parameters of an algorithm using a random search

























