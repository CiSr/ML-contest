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


print "...loading the training data file"
my_data = genfromtxt('contest_train.csv', delimiter=',')
train_data=my_data[:,0:10]
label_data=my_data[:,1897]


#Filling the missing values
x_temp=sp.csc_matrix(train_data)
#imp.fit(train_data)
imp =Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit(train_data)
inter=imp.transform(x_temp)
print "\nThe preprocessed input values are:"
print "\n"
print inter

X_new = SelectKBest(chi2, k=10).fit_transform(inter, label_data)

print "\n...preprocessed dataset are: "
print "\n"
print X_new.shape



#loading the test set with missing values being filled
test_data=genfromtxt('contest_only_test.csv',delimiter=',')
test_data=test_data[:,0:10]
#print test_data
inter_test=sp.csc_matrix(test_data)
imp.fit(inter_test)
test_final=imp.transform(inter_test)
print "\nThe preprocessed test values are:"
print "\n"
print test_data

#if(test_final==test_data):
#       print "yes"
#else:
#       print "no"
#print label_data.shape
print train_data.shape
print test_data.shape

X = np.array(inter)
Y = np.array(label_data)
# Setting the perceptron





clf=PCA(n_components=10)
#clf = RandomForestClassifier(n_estimators=3)
print clf.fit(X)
#intermediate=clf.predict(test_final)
clf.explained_variance_ratio_
#termin*np.transpose(inter)
print "\n"
temper=clf.transform(X)

print temper.shape
final_preprocess=temper*inter

hat=np.array(final_preprocess)
#init=RandomForestClassifier(n_estimators=3)
init=SVC()
print init.fit(hat,Y)
intra=init.predict(test_final)

print intra
print "score prediction"
comp=np.ones(1500)
top_k=init.score(test_final,comp)


print "shape of score :"
print top_k.shape
#print accuracy_score(test_final,intermediate)
#for i in range(len(intra)):
#	print intra[i]
 #	print "\n"
#print intra.shape

#print(clf.decision_function(test_final))
plt.plot(final_preprocess,'x');plt.axis('equal');plt.show()


