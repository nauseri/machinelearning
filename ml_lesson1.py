

# ml mastery python ml mini-course lessons 1-6:

# Lesson 1: Download and Install Python and SciPy
# install everything at once (much easier) with Anaconda.

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


# Lesson 2: Get Around In Python, NumPy, Matplotlib and Pandas.

# Practice assignment, working with lists and flow control in Python.
# goal, practice basic python programming language syntax and imp. scipy data structures

# Practice working with NumPy arrays.
# a = np.array([1,2,3])
# print(a)
# b = np.array([[1, 2], [3, 4]])
# print(b)
# dt = np.dtype([('age',np.int8)])
# c = np.array([(10,),(20,),(30,)], dtype = dt)
# print(c)
# print(c['age'])
# student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')])
# d = np.array([('mark', 21, 50),('bono', 18, 75)], dtype = student)
# print(d)
#
# print(b.shape)
# e = np.array([[1,2,3],[4,5,6]])
# e.shape = (3,2)
# print(e)
# a = np.arange(24)
# a.ndim
# b = a.reshape(2,4,3)
# print(b)
# print(b.itemsize)  # returns the length of each element of array in bytes
# x = [1,2,3]
# a = np.asarray(x)
# print(a)

# # Practice creating simple plots in Matplotlib.
# import matplotlib.pyplot as plt
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# plt.ylabel('some numbers')
# plt.xlabel('some values')
# plt.show()
#
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.show()
#
# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)  # numpy.arange(start, stop, step, dtype)
# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()
#
# data = {'a': np.arange(50),
#         'c': np.random.randint(0, 50, 50),
#         'd': np.random.randn(50)}
# data['b'] = data['a'] + 10 * np.random.randn(50)
# data['d'] = np.abs(data['d']) * 100
#
# plt.scatter('a', 'b', c='c', s='d', data=data)
# plt.xlabel('entry a')
# plt.ylabel('entry b')
# plt.show()
#
# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)
#
# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
#
# plt.figure()
# plt.subplot(211)
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
#
# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# plt.show()
#
# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
#
# # the histogram of the data
# n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
#
#
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)
# plt.show()

# Practice working with Pandas Series and DataFrames.
# s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])  # in pandas, is a series, 1 dim array
# print(s)
# print(s.index)
# # Index(['a'/, 'b', 'c', 'd', 'e'], dtype='object')
# print(pd.Series(np.random.randn(5)))
# d = {'b': 1, 'a': 0, 'c': 2}
# print(pd.Series(d))
# # DataFrame is a 2-dim labeled data structure with columns of potentially different types
# d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
# df = pd.DataFrame(d)
# print(df)
# print(pd.DataFrame(d, index=['d', 'b', 'a']))
# print(pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three']))
# print(df['one'])

# myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
# rownames = ['a', 'b']
# colnames = ['one', 'two', 'three']
# mydf = pandas.DataFrame(myarray, index=rownames, columns=colnames)
# print(mydf)


# Lesson 3: Load Data From CSV
# goal practice, get comfy w data loading into python of standard ml datasets
#
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# independant variables;
# pregnancies, plasma glucose, blood pressure, skin thickness, insulin,
# BMI, diabetes pedigree function, age
# dependant variable: outcome
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
mydata = pd.read_csv(url, names=header)
print(mydata.shape)
print(mydata)


# Lesson 4: Understand Data with Descriptive Statistics
# goal practice using descriptive statistics to understand data, use pandas dataframe helper functions

# Statistical Summary
# description = mydata.describe()
# print(description)
# print(mydata.head(5))
# print(mydata.shape)
# print(mydata.dtypes)
# print(mydata.corr())


# Lesson 5: Understand Data with Visualization
# goal practice plotting in python, understand attributes alone and together,
# use pandas dataframe helper functions

# Scatter Plot Matrix
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pandas.plotting import hist_frame
from pandas.plotting import boxplot
# scatter_matrix(mydata)
# plt.show()
# hist_frame(mydata, bins=10)
# plt.xlabel(("nr"))
# plt.ylabel("frec")
# plt.show()
# boxplot(mydata)
# plt.show()


# Lesson 6: Prepare For Modeling by Pre-Processing Data
# goal practice scikit-learn library to transform data

# Standardize data (mean 0 and stdev 1)
# using the scale and center options.
from sklearn.preprocessing import StandardScaler
array = mydata.values
# separate array into input and output components
X = array[:, 0:8]  # all rows, columns from 0 to 7, are indep. var.
Y = array[:, 8]  # all rows, only column 8, is dep. var.
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
# print(rescaledX[0:5, :])
rescaledXX = pd.DataFrame(rescaledX)
description1 = rescaledXX.describe()
print(description1)
boxplot(rescaledXX)
plt.show()

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
boxplot(mm_rescaledXX)
plt.show()

# Explore more advanced feature engineering such as Binarizing




