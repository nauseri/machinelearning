
# FINAL, as seen in ml mastery w python: chp 19


import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import xgboost
import matplotlib.pyplot as plt
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from pickle import dump
from pickle import load


# https://archive.ics.uci.edu/ml/datasets/Iris
# Attribute Info: 1. sepal length in cm, 2. sepal width in cm, 3. petal length in cm, 4. petal width in cm,
# 5. class: Iris Setosa, Iris Versicolour, Iris Virginica
filename = 'iris.data.csv'
names = ['sepal len', 'sepal wid', 'petal len', 'petal wid', 'class']
data = read_csv(filename, names=names)
# Split-out validation dataset
array = data.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Spot-Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier()))          # 2
models.append(('LDA', LinearDiscriminantAnalysis()))    # 3
models.append(('rf', RandomForestClassifier(n_estimators=100)))
models.append(('SVM', SVC(gamma='auto')))               # 1
models.append(('xgb', xgboost.XGBClassifier()))         # 3
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset: svc
svc = SVC(gamma='auto')
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print('svc acc ', accuracy_score(Y_validation, predictions)*100)
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset: knn
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions1 = knn.predict(X_validation)
print('knn acc ', accuracy_score(Y_validation, predictions1)*100)
print(confusion_matrix(Y_validation, predictions1))
print(classification_report(Y_validation, predictions1))


# standardize data
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
pipelines.append(('scaledKNN', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('scaledLDA', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('scaledRF', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('RFC', RandomForestClassifier(n_estimators=num_trees, max_features=max_features, random_state=seed))])))
pipelines.append(('scaledSVC', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('SVM', SVC(gamma='auto'))])))
pipelines.append(('scaledXGB', Pipeline([('feature_union', feature_union), ('StandardScale',  StandardScaler()), ('XGB', xgboost.XGBClassifier(max_depth=3, eta=1, objective='reg:logistic'))])))

# evaluate pipeline
num_folds = 10
scoring = 'accuracy'
results1 = []
names1 = []
for name1, model1 in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results1 = cross_val_score(model1, X_train, Y_train, cv=kfold, scoring=scoring)
    results1.append(cv_results1)
    names1.append(name1)
    msg = "%s: %f (%f)" % (name1, cv_results1.mean()*100, cv_results1.std()*100)
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results1)
ax.set_xticklabels(names1)
plt.show()








