import csv
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import scipy.sparse as sp
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import KFold
import random 
import matplotlib.pyplot as plt
print "..loading the training data"

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






my_data=genfromtxt('table.csv',delimiter=',')
train_set=my_data[:,1:5];
test_set=my_data[:,6];
count=2000;





#clf=PCA(n_components=1)
clf=PCA(n_components=1)
clf.fit_transform
clf.fit_transform(train_set)
temper=clf.transform(train_set)
temper
abs(temper)
temper=abs(temper)

plt.plot(temper)
plt.title('Close feature')
plt.show()
#plt.show()
#plt.plot(temper)
#plt.show()
#plt.title('Closed feature')

temper.shape
for i in range(3473):
        print temper[i]

nter=train_set[1:1000,3];
plt.plot(nter)
plt.title('CLosed data')
plt.show()
#np.array(inter)
#Y = np.array(label_data)
# Setting the perceptron





#clf=PCA(n_components=10)
#clf = RandomForestClassifier(n_estimators=3)
#print clf.fit(X
