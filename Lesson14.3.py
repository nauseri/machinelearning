
import sys
import numpy as np
import pandas as pd
import sklearn  #scikit-learn
from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import xgboost  # at eth computer, need to add this with conda
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from pandas import set_option


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

# Scale numerical data (w range 0-1) w range option
# mm_scale = StandardScaler().fit(X)
mm_scale = MinMaxScaler().fit(X)
mm_rescaled = mm_scale.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
Xre = pd.DataFrame(mm_rescaled)

# Evaluation:
test_size = 0.3
seed = 7
Y_test: object
X_train, X_test, Y_train, Y_test = train_test_split(Xre, Y, test_size=test_size, random_state=seed)
kfold = KFold(n_splits=20, random_state=seed)

# Evaluate using kfold Cross Validation
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
report = classification_report(Y_test, predicted)
print('lda\n', report)

model1 = RandomForestClassifier(n_estimators=100, random_state=seed)
model1.fit(X_train, Y_train)
predicted = model1.predict(X_test)
report = classification_report(Y_test, predicted)
print('rf\n', report)

model2 = xgboost.XGBClassifier(max_depth=3, eta=1, objective='reg:logistic')
model2.fit(X_train, Y_train)
predicted = model2.predict(X_test)
report = classification_report(Y_test, predicted)
print('xgb\n', report)

# Compare Machine Learning Algorithms
models = []
models.append(('RFC', RandomForestClassifier(n_estimators=100, random_state=seed)))
models.append(('XGB', xgboost.XGBClassifier(max_depth=3, eta=1, objective='reg:logistic')))
models.append(('SVM', SVC(gamma='auto')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
#
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    set_option('precision', 3)
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()







