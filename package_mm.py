import csv
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import scipy.sparse as sp
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import Perceptron






my_data=genfromtxt('table.csv',delimiter=',')
train_set=my_data[:,1:5];
test_set=my_data[:,6];
inter_test=np.ones(3473,1)
count=2000;
print inter_test.shape

clf=Perceptron();
clf.fit(train_set,inter_test);

clf.test(test_Set);

pk_normal =write(test_set)









