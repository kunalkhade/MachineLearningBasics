'''
    File name: test.py
    Supporting file: ML.py
    Author: Kunal Khade
    Date created: 9/20/2020
    Date last modified: 9/24/2020
    Date last modified: 10/30/2020
    Python Version: 3.7

    Topic 1: Develop generic binary classifier perceptron 
    class in ML.py.  It has to taketraining  set  of  any  size.   
    Class  must  include  four  functions  :init(),  fit()  ,netinput(), 
    predict(), One more supportive function to display result.


    Topic 2: Develop Linear Regression classifier using perceptron model 
    class Linear_Regression in ML.py.  It has to taketraining  set  of  any  size.   
    Class  must  include single function call Linear_Regression(A, B) and pass
    A = Learning Rate (0.01) and B = Iterations(5-10000). Call function final run and pass
    points [X, Y] array format. It will display result. 
    
'''
from ML import Perceptron
from ML import Linear_Regression
from ML import Logistic_Regression
from ML import Decision_Stump
from ML import Intervals
from ML import SVM
from ML import knn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import ListedColormap

pn = Perceptron(0.1, 10)
lr = Linear_Regression(0.01, 50)
log_r = Logistic_Regression(0.1, 1000)
dec_s = Decision_Stump()
Int = Intervals()
svm = SVM()
knn = knn()
#Using Pandas import Iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print("-----------------------------------------------------------------------------------------")
print("SUPPORT VECTOR MACHINE - ")
X_1 = df.iloc[0:50, [1,2]].values
X_2 = df.iloc[51:100, [1,2]].values                                              ##SVM
svm.data_handle(X_1, X_2)
svm.SVM_fit()
svm.visualize()


print("---------------------------------------------------------------------------------------------------")
print("PERCEPTRON MODEL - ")
#Only use initial 100 data value labels 
y = df.iloc[0:100, 4].values
#Convert labels into -1 and 1
y = np.where(y == 'Iris-setosa', -1, 1)
#Extract only 2 parameters from data set
X = df.iloc[0:100, [0, 2]].values
#Use fit, error, predict, weights, net_input functions from perceptron class
pn.fit(X, y)                                                                    ##PERCEPTRON
print("Errors : \n", pn.error)
print("Prediction : \n",pn.predict(X)) 
print("Weights : \n", pn.weights)
print(pn.net_input(X))
#Plot result 
pn.plot_decision_regions(X, y, classifier=pn, resolution=0.02)
#Plot Error function 
plt.plot(range(1, len(pn.error) + 1), pn.error, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Misclassifications')
plt.title('Error function')
plt.show()

print("---------------------------------------------------------------------------------------------------")
print("LINEAR REGRESSION - ")
x1 = df.iloc[0:50, 1].values            #Linear Regression
y1 = df.iloc[0:50, 0].values
points = np.stack((x1,y1),axis=1)
lr.final_run(points)                                        ##Linear REGRESSION

print("---------------------------------------------------------------------------------------------------")
print("LOGISTIC REGRESSION - ")
#Adjusting Data
X_setosa_train = df.iloc[0:40, 1].values
Y_setosa_train = df.iloc[0:40, 3].values
X_setosa_test = df.iloc[41:50, 1].values
Y_setosa_test = df.iloc[41:50, 3].values
X_Versicolor_train = df.iloc[50:90, 1].values
Y_Versicolor_train = df.iloc[50:90, 3].values
X_Versicolor_test = df.iloc[91:100, 1].values
Y_Versicolor_test = df.iloc[91:100, 3].values
X_total = df.iloc[0:100, [1,3]].values
X_Update_Train = np.concatenate((X_setosa_train, X_Versicolor_train),axis=0)
Y_Update_Train = np.concatenate((Y_setosa_train, Y_Versicolor_train),axis=0)
X_Update_Test = np.concatenate((X_setosa_test, X_Versicolor_test),axis=0)
Y_Update_Test = np.concatenate((Y_setosa_test, Y_Versicolor_test),axis=0)
X_train = np.stack((X_Update_Train,Y_Update_Train),axis=1)
X_test = np.stack((X_Update_Test,Y_Update_Test),axis=1)
y1 = df.iloc[0:40, 4].values
y2 = df.iloc[50:90, 4].values
y3 = df.iloc[41:50, 4].values
y4 = df.iloc[91:100, 4].values
Y_train = np.concatenate((y1, y2),axis=0) 
y_train = np.where(Y_train == 'Iris-setosa', 0, 1)
Y_test = np.concatenate((y3, y4),axis=0) 
Y_test = np.where(Y_test == 'Iris-setosa', 0, 1)
Y_total = df.iloc[0:100, 4].values
Y_total = np.where(Y_total == 'Iris-setosa', 0, 1)
log_r.Logistic_Regression(X_train,y_train,X_test,Y_test,X_total,Y_total)            ##LOG REGRESSION

print("---------------------------------------------------------------------------------------------------")
print("LOW VC DIMENSION DECISION STUMP - ")
X = df.iloc[0:100, [1,3]].values
Y_train = df.iloc[0:100, 4].values
Y = np.where(Y_train == 'Iris-setosa', -1, 1)
dec_s.buildStump(X,y)                                                               ##DECISION STUMP
print("---------------------------------------------------------------------------------------------------")
print("LOW VC DIMENSION INTERVALS - ")
Int.intervals(x1)                                                                   ##DECISION STUMP       
print("---------------------------------------------------------------------------------------------------")
print("1 Nearest Neighbour - ")

X_setosa_train = df.iloc[0:40, 1].values
Y_setosa_train = df.iloc[0:40, 3].values
 
X_setosa_test = df.iloc[41:50, 1].values
Y_setosa_test = df.iloc[41:50, 3].values

X_Versicolor_train = df.iloc[50:90, 1].values
Y_Versicolor_train = df.iloc[50:90, 3].values                                         ##KNN

X_Versicolor_test = df.iloc[91:100, 1].values
Y_Versicolor_test = df.iloc[91:100, 3].values

knn.knn_process(X_setosa_train, X_Versicolor_train, Y_setosa_train, Y_Versicolor_train, X_setosa_test, Y_setosa_test)

print("---------------------------------------------------------------------------------------------------")
print("Thank You ")