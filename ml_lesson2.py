
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


# Lesson 7: Algorithm Evaluation With Resampling Methods
# goal practice resampling methods in scikit-learn

# resampling: statistical methods split training dataset into subsets
# some used to train model and others used to estimate accuracy of model on unseen data
from pandas import read_csv
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Scale numerical data (w range 0-1) w range option
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler().fit(X)
mm_rescaledX = mm_scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
# print(mm_rescaledX[0:5, :])
mm_rescaledXX = pd.DataFrame(mm_rescaledX)
description2 = mm_rescaledXX.describe()
print(description2)


# Split a dataset into training and test sets
from sklearn.model_selection import train_test_split
# Split arrays or matrices into random train and test subsets
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(mm_rescaledXX, Y, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Estimate the accuracy of an algorithm using k-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# K-Folds cross-validator: provides train/test indices to split data in train/test sets.
# Split dataset into k consecutive folds (without shuffling by default).
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
kfold = KFold(n_splits=10, random_state=7)
# Logistic Regression (aka logit, MaxEnt) classifier.
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
model = LogisticRegression(solver='liblinear')
# Evaluate a score by cross-validation
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
results = cross_val_score(model, mm_rescaledXX, Y, cv=kfold)
print("Accuracy logreg: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

# linear svc, support vector machine...
from sklearn.svm import LinearSVC
# X, y = make_classification(n_features=8, random_state=0)
kfold2 = KFold(n_splits=20, random_state=7)
model2 = LinearSVC(random_state=0, tol=1e-5)
results2 = cross_val_score(model2, mm_rescaledXX, Y, cv=kfold2)
print("Accuracy linsvc: %.3f%% (%.3f%%)" % (results2.mean()*100.0, results2.std()*100.0))

# # Estimate the accuracy of an algorithm using leave one out cross validation
# from sklearn.model_selection import LeaveOneOut
# # LeaveOneOut (or LOO) is a simple cross-validation. Each learning set created by taking
# # all the samples except one, test set being the sample left out. for samples,
# # we have different training sets and  different tests set
# # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut
# loo = LeaveOneOut()
# loo.get_n_splits(X)
# for train, test in loo.split(X):
#     print("%s %s" % (train, test))







