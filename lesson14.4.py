

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



# https://archive.ics.uci.edu/ml/datasets/Iris
# Attribute Info: 1. sepal length in cm, 2. sepal width in cm, 3. petal length in cm, 4. petal width in cm,
# 5. class: Iris Setosa, Iris Versicolour, Iris Virginica
filename = 'iris.data.csv'
names = ['sepal len', 'sepal wid', 'petal len', 'petal wid', 'class']
data = read_csv(filename, names=names)
del data['sepal wid']
array = data.values
X = array[:, 0:3]
Y = array[:, 3]
target_names = ['Iris-setosa', 'Iris-versicolour', 'Iris-virginica']
test_size = 0.3
seed = 7



# create feature union
features = []
features.append(('PCA', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=3)))
feature_union = FeatureUnion(features)
# create pipeline
estimator = []
estimator.append(('feature_union', feature_union))
estimator.append(('MinMaxScale',  MinMaxScaler()))

estimator_mid = Pipeline(estimator)

estimators = []
model = Pipeline(estimators)


estimators.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))

# estimators.append(('RFC', RandomForestClassifier(n_estimators=100, random_state=seed)))
# estimators.append(('XGB', xgboost.XGBClassifier(max_depth=3, eta=1, objective='reg:logistic')))
# estimators.append(('SVM', SVC(gamma='auto')))
# estimators.append(('LDA', LinearDiscriminantAnalysis()))
# estimators.append(('CART', DecisionTreeClassifier()))

scoring = 'accuracy'
kfold = KFold(n_splits=10, random_state=7)

for i in estimators:
    # evaluate pipeline
    model = Pipeline(estimators)
    # with CV, need only training and test set (only 2 partitions of the data)
    # no validation set needed as in split_train_test (3 partitions of the data)
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    print("accuracy %f (%f)" % (results.mean()*100, results.std()*100))

























#
# # Evaluation:
# test_size = 0.3
# seed = 7
# Y_test: object
# X_train, X_test, Y_train, Y_test = train_test_split(Xre, Y, test_size=test_size, random_state=seed)
# kfold = KFold(n_splits=20, random_state=seed)
#
# # Evaluate using kfold Cross Validation
# model = LinearDiscriminantAnalysis()
# model.fit(X_train, Y_train)
# predicted = model.predict(X_test)
# matrix = confusion_matrix(Y_test, predicted)
# print(matrix)
# # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
# report = classification_report(Y_test, predicted)
# print('lda\n', report)
#
# model1 = RandomForestClassifier(n_estimators=100, random_state=seed)
# model1.fit(X_train, Y_train)
# predicted = model1.predict(X_test)
# report = classification_report(Y_test, predicted)
# print('rf\n', report)
#
# model2 = xgboost.XGBClassifier(max_depth=3, eta=1, objective='reg:logistic')
# model2.fit(X_train, Y_train)
# predicted = model2.predict(X_test)
# report = classification_report(Y_test, predicted)
# print('xgb\n', report)
#
# # Compare Machine Learning Algorithms
# models = []
#
# models.append(('RFC', RandomForestClassifier(n_estimators=100, random_state=seed)))
# models.append(('XGB', xgboost.XGBClassifier(max_depth=3, eta=1, objective='reg:logistic')))
# models.append(('SVM', SVC(gamma='auto')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('CART', DecisionTreeClassifier()))
# #
# results = []
# names = []
# scoring = 'accuracy'
#
# for name, model in models:
#     set_option('precision', 3)
#     kfold = KFold(n_splits=10, random_state=7)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
#     print(msg)
#
# # boxplot algorithm comparison
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()
#
