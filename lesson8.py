
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


# Lesson 8: Algorithm Evaluation Metrics:
# specify the metric used for your test harness in scikit-learn via
# cross_validation.cross_val_score() function and defaults can be used for regression and classification problems
# goal practice using different algo performance metrics in the scikit-learn

# Practice using the Accuracy and LogLoss metrics on a classification problem
# Cross Validation Classification LogLoss
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
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
np.set_printoptions(precision=2)
# print(mm_rescaledX[0:5, :])
rescaled_X = pd.DataFrame(mm_rescaledX)
description2 = rescaled_X.describe()
print(description2)

kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
scoring = 'neg_log_loss'
results = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
scoring = 'accuracy'
results = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring=scoring)  # dif scoring types, see
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print("Accuracy: %.3f (%.3f)" % (results.mean()*100, results.std()*100))

# Split a dataset into training and test sets
from sklearn.model_selection import train_test_split
# Split arrays or matrices into random train and test subsets
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(rescaled_X, Y, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test.shape)
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix


# Plot normalized confusion matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

# print(y_test.shape)  # both are numpy.ndarray
# print(y_test.dtype)  # float64
# print(y_pred.shape)
# print(y_pred.dtype)  # float64
# y_pred = y_pred.astype('int8')
# print(y_pred.dtype)
# y_test = y_test.astype('int8')
# print(y_pred.dtype)
# print(y_test)
# print(y_pred)
# y_y_test = np.array(y_test)
# y_pred = np.array(y_pred)

header = names

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = list(unique_labels(y_true, y_pred))  # https://stackoverflow.com/questions/46902367/numpy-array-typeerror-only-integer-scalar-arrays-can-be-converted-to-a-scalar-i
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=header, normalize=True, title='Normalized confusion matrix')
plt.show()

from sklearn.metrics import classification_report  # https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# Practice using RMSE and RSquared metrics on a regression problem













