#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 23:11:05 2018

@author: ajinkya_1610
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as t

data = pd.read_csv("Churn_Modelling.csv")
x = data.iloc[:,3:13].values     #all the columns except the last one is considered
y = data.iloc[:,13].values

#label encoding
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as ohe
le_x_1 = le()             #label encoder object created for country
x[:,1]=le_x_1.fit_transform(x[: ,1]) #label encoder object linked with the 2nd column of the data table
le_x_2 = le()             #label encoder object created for gender
x[:,2]=le_x_2.fit_transform(x[: ,2])
ohec=ohe(categorical_features=[1]) #index of the column is to be specified for the onehot encoding
x = ohec.fit_transform(x).toarray() #now we have to fit the ohec object into
x=x[:,1:]                           #to eliminate the dummy variable trap(like for three classes a dummy variable set of 2 is fine(third is automatically set))

#data splitting
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler as sc
sc_x= sc()
x_train = sc_x.fit_transform(x_train)    #standardization scaling we are doing    
x_test = sc_x.transform(x_test) 

import keras as ke 
from keras.models import Sequential #to initialize the ann
from keras.layers import Dense      #to build the layers of the ann

#INITIALIZING THE ANN
classifier=Sequential()             #sequential object created( ann as sequence of layers)

#adding the input layer and first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))           #now we will give the no of nodes in hidden layer, activation function=rectifier function,initialized value for the weights=uniform distribution
'''we generally prefer the rectfier function as the activation function in case of the hidden layer activation and sigmoid function for the 
output layer activation'''
#now we need to add the sencond hidden layer
classifier.add(Dense(activation="relu",units=6, kernel_initializer="uniform"))           #now for the second hidden layer no need of input dimensions as first hidden layer is already created so the output of that layer is the input to this layer(so input_dim=11)
'''now the dimension of the output layer for 2nd hidden layer is the 2nd hidden layer itself.
so we will better use the dimension to be average of the input and the output layer dimensions'''

#now we will add the the output layer
classifier.add(Dense(activation="sigmoid", units=1,kernel_initializer="uniform"))       #activ_func=suftmax(or somethong like that) if we have output in multiple categories
'''output has the sigmoid function activation'''

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

'''optimizer is the algorithm that we use for finding the parameters- i.e stochastic
grdient descent(named as adam by the specifier. loss is the error function for stochastic 
gradient descent=(1/m)summation(-y(i)*log(htheta(x))-(1-y(i))*log(1-htheta(x(i)))).
Metrices is the parameter the compile method should focus on. i.e. implementation objective here
is to increase the accuracy'''
t1=t.time()
#Fitting the ANN to the training set
classifier.fit(x_train, y_train,batch_size=10,epochs=100)
t2=t.time()
print('Fitting took'+str(t2-t1)+'seconds')
'''x_train is the data which we want to train the model
y_train is the targeted labels which the model will try to
achieve. batch is the size of batch after which the model will 
update the weights. epochs is the number of times we will be redoing the entire
procedure for better accuracy'''

#predicting the Test set results
y_pred=classifier.predict(x_test)
y_pred=(y_pred > 0.5)
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

'''After this we will try to find out the accuracy of our model on the unknown data
i.e.test data....
from the confusion_matrix we get the total number of correct predictions=True positives+True Negatives
Accuracy=net correct predictions/total test data set size
'''

'''Try with a new customer'''

inp=[[0,0,600,1,40,3,60000,2,1,1,50000]]
prediction=classifier.predict(sc_x.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
pred=(prediction>0.5)
print(pred)

'''the explained method in the solution-
new_prediction=classifier.predict(np.array([[info of the new customer]]))'''

