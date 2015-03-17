import csv
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
#imp=Imputer(missing_values='NaN',strategy='mean')
import scipy.sparse as sp
import numpy as np
from sklearn.linear_model import Perceptron

my_data = genfromtxt('contest_train.csv', delimiter=',')
train_data=my_data[:,1:1896]
label_data=my_data[:,1897]
x_temp=sp.csc_matrix(train_data)
#imp.fit(train_data)
imp =Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit(train_data)
inter=imp.transform(x_temp)
print inter

test_data=genfromtxt('contest_only_test.csv',delimiter=',')
print test_data
inter_test=sp.csc_matrix(test_data)
imp.fit(inter_test)
test_final=imp.transform(inter_test)
#print test_data


#if(test_final==test_data):
#       print "yes"
#else:
#       print "no"
print label_data.shape
print inter.shape
print test_final.shape

X = np.array(inter)
Y = np.array(label_data)
# Setting the perceptron
clf = Perceptron()
#Fit the perceptron to the data
clf.fit(X, Y)
#clf.predict(test_final)
