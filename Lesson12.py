
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

# Lesson 12: Improve Accuracy with Ensemble Predictions
# improve model by combining models
# for some models this is build in: random forest for bagging and stochastic gradient boosting for boosting
# Another type of ensembling called voting can be used to combine the predictions from multiple different models together

# https://scikit-learn.org/stable/modules/ensemble.html#bagging

# goal practice ensemble methods
# Practice bagging ensembles with the random forest (and extra trees algorithms)
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
clf1 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)  # Random Forest algorithm (a bagged ensemble of decision trees)
results = cross_val_score(clf1, X, Y, cv=kfold)
print('RandFor: cross_val_score mean: ', results.mean())


# Practice boosting ensembles with AdaBoost (and gradient boosting machine) algorithms
from sklearn.ensemble import AdaBoostClassifier
clf2 = AdaBoostClassifier(n_estimators=20)
scores = cross_val_score(clf2, X, Y, cv=kfold)
print('AdaBoost: cross_val_score mean: ', scores.mean())

# Practice voting ensembles using by combining the predictions from multiple models together
# https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('rf', clf1), ('ab', clf2)], voting='hard')
for clf, label in zip([clf1, clf2, eclf], ['Random Forest', 'AdaBoost', 'Ensemble']):
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))













