
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import xgboost
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from pandas import set_option
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
from sklearn.externals import joblib


# https://archive.ics.uci.edu/ml/datasets/Iris
# Attribute Info: 1. sepal length in cm, 2. sepal width in cm, 3. petal length in cm, 4. petal width in cm,
# 5. class: Iris Setosa, Iris Versicolour, Iris Virginica
filename = 'iris.data.csv'
names = ['sepal len', 'sepal wid', 'petal len', 'petal wid', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, 0:4]
Y = array[:, 4]
target_names = ['Iris-setosa', 'Iris-versicolour', 'Iris-virginica']

# Scale numerical data (0-1)
ss_scale = StandardScaler().fit(X)
ss_rescaled = ss_scale.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
Xre = pd.DataFrame(ss_rescaled)
test_size = 0.3
seed = 7
Y_test: object
X_train, X_test, Y_train, Y_test = train_test_split(Xre, Y, test_size=test_size, random_state=seed)

# Fit the model on 33%
# model = RandomForestClassifier(n_estimators=107)
# model = SVC(gamma='auto')
model = LinearDiscriminantAnalysis()

model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model_14.7.sav'
dump(model, open(filename, 'wb'))

# some time later...
# load the model from disk
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)


# similar as above but with joblib
# # save the model to disk
# filename = 'finalized_model.sa'
# joblib.dump(model, filename)
# # some time later...
# # load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, Y_test)
# print(result)








