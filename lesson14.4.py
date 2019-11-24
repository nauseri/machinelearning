

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
from sklearn.neighbors import KNeighborsClassifier

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

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7
num_trees = 100
max_features = 3
# create feature union
features = []
features.append(('PCA', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=3)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('StandardScaler',  StandardScaler()))
pipelines = []
pipelines.append(('scaledRF', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('RFC', RandomForestClassifier(n_estimators=num_trees, max_features=max_features, random_state=seed))])))
pipelines.append(('scaledXGB', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('XGB', xgboost.XGBClassifier(max_depth=3, eta=1, objective='reg:logistic'))])))
pipelines.append(('scaledSVC', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('SVM', SVC(gamma='auto'))])))
pipelines.append(('scaledLDA', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('scaledKNN', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('KNN', KNeighborsClassifier())])))

# evaluate pipeline
num_folds = 10
scoring = 'accuracy'
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
    print(msg)




