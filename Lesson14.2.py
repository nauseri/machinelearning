# practice putting the pieces together and working through a standard machine learning dataset en d -t o -end
# Work through the iris dataset end-to-end

# steps:
# Understanding your data using descriptive statistics and visualization.
# Preprocessing the data to best expose the structure of the problem.
# Spot-checking a number of algorithms using your own test harness.
# Improving results using algorithm parameter tuning.
# Improving results using ensemble methods.
# Finalize the model ready for future use.

import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

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

# Scale numerical data (w range 0-1) w range option
mm_scale = MinMaxScaler().fit(X)
mm_rescaled = mm_scale.transform(X)
# summarize transformed data
Xre = pd.DataFrame(mm_rescaled)
np.set_printoptions(precision=3)
descript = Xre.describe()
print('description\n', descript)
Frame = pd.DataFrame(Xre.values, columns=['sepal len', 'sepal wid', 'petal len', 'petal wid'])
Frame.plot(kind='box', subplots=False, layout=(3, 3), sharex=False, sharey=False)
plt.show()

# feature extraction w US Univariate Selection
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print('US ', fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :])  # highest scores are best features to use, so
#sepal length, petal length, petal width

# feature extraction w RFE Recursive Feature Elimination
model = LinearSVC()
rfe = RFE(model, 3)
fit2 = rfe.fit(Xre, Y)  # True or 1 as best features
print("RFE: Num Features: %d" % fit2.n_features_)
print("Selected Features: %s" % fit2.support_)
print("Feature Ranking: %s" % fit2.ranking_)

# feature extraction w PCA Principal Component Analysis
iris = datasets.load_iris()
Y2 = iris.target
target_names = ['Iris-setosa', 'Iris-versicolour', 'Iris-virginica']
pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)
# Percentage of variance explained for each components
print('PCA: explained variance ratio (first 3 components): %s' % str(pca.explained_variance_ratio_))
print(pca.components_)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[Y2 == i, 0], X_r[Y2 == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.figure()
plt.show()

# feature extraction w FI Feature Importance
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, Y)
print('FI ', model.feature_importances_)  # the larger the score, the more important the attribute

# best 3 features
# US 1, 3, 4
# RFE 2, 3, 4
# FI 1, 3, 4
# choose 1, 3, 4

















# test_size = 0.33
# seed = 7
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# # Fit the model on 33%
# model = LogisticRegression(solver='liblinear')
# model.fit(X_train, Y_train)
# # save the model to disk
# filename1 = 'final_model.sav'
# pickle.dump(model, open(filename1, 'wb'))  # pickle framework for saving ml models for later use, such as saving a final model
#
# # some time later...
#
# # load the model from disk
# load_model = pickle.load(open(filename1, 'rb'))
# result1 = load_model.score(X_test, Y_test)
# print(result1)













