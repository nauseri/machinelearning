
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



# https://archive.ics.uci.edu/ml/datasets/Iris
# Attribute Info: 1. sepal length in cm, 2. sepal width in cm, 3. petal length in cm, 4. petal width in cm,
# 5. class: Iris Setosa, Iris Versicolour, Iris Virginica
filename = 'iris.data.csv'
names = ['sepal len', 'sepal wid', 'petal len', 'petal wid', 'class']
data = read_csv(filename, names=names)
# del data['sepal wid']
array = data.values
X = array[:, 0:4]
Y = array[:, 4]
target_names = ['Iris-setosa', 'Iris-versicolour', 'Iris-virginica']
test_size = 0.3
seed = 7
num_trees = 100
max_features = 3

# # create feature union
features = []
features.append(('PCA', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=3)))
feature_union = FeatureUnion(features)
# create pipeline and sub models
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('StandardScaler',  StandardScaler()))


# model = LinearDiscriminantAnalysis()  # lda and svc best models so far...
# estimators.append(('lda', model))
# model = SVC(gamma='auto')
# estimators.append(('svm', model))
# model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, random_state=seed)
model = RandomForestClassifier(n_estimators=20)
estimators.append(('rfc', model))
# model = xgboost.XGBClassifier(max_depth=3, eta=1, objective='reg:logistic')
# estimators.append(('xgb', model))
param_grid = {'n_estimators': [105, 106, 107]}
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=3, random_state=7, iid=False)
rsearch.fit(X, Y)
print(rsearch.best_estimator_)
print(rsearch.best_index_)
print(rsearch.best_params_)

# cross validation
model = Pipeline(estimators)
scoring = 'accuracy'
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("accuracy %f (%f)" % (results.mean()*100, results.std()*100))










