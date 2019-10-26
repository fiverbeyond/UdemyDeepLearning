# -*- coding: utf-8 -*-
"""
Introduction file for Udemy's Deep Learning course.

This file includes my personal solution to the homework (i.e., predict whether a single
specified customer was likely to leave the bank.) My solution to this question was much more
complicated than the solution offered in the homework solution file (in this folder); I could
have simply applied the feature selection scaling to a hard-coded matrix, rather than
mirror the entire scaling and selection process starting from a one-line CSV file.

Library Review:
    
    Theano - An open source efficient numeric computations library, based on numpy syntax. Useful, in that it can run on your GPU.
    
    TensorFlow - Used professionally (and in academia) to build neural networks from scratch.
    
    Keras - Based on Theano and TensorFlow. Simplifies the building of deep learning models.

"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas

# Importing the dataset
dataset = pandas.read_csv('Churn_Modelling.csv')
# We select only a subset of columns, as many of them can be dismissed as irrelevant.
X = dataset.iloc[:, 3 : 13].values # Upper-bound excluded, so marking 13 takes up to row 12.
y = dataset.iloc[:, 13].values

# Encode categorical data to numerical values
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()  # For encoding the 'Country' column.
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # 1, the 'Country' column.)
labelencoder_X_2 = LabelEncoder()  # For encoding the 'Gender' column.
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


# We only need to create dummy variables for the first column, as there are many countries.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap, which in this case, only applies to the Country column
# which at this point has been split into three (exclusive) columns.
X = X[:, 1:]  # Remove the first column.
homeworkCustomer = homeworkCustomer[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###   BUILDING THE ARTIFICIAL NEURAL NETWORK     ###

#Importing the Keras libraries and packages.
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialize the ANN
myNeuralNetwork = Sequential()


# Fitting classifier to the Training set


# Adding the first hidden layer (and specifying the input layer, since this is the firsts
# hidden layer we're adding.)
# 6 nodes, uniform weights, rectifier activation function ('relu')
# With dropout.
myNeuralNetwork.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
myNeuralNetwork.add(Dropout(p = 0.1))

# adding a second hidden input layer. Not necessary for this example, but good practice.
myNeuralNetwork.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
myNeuralNetwork.add(Dropout(p = 0.1))

# Adding the output layer.
# This only has one output node, as this is a binary classifier. We use a sigmoid function for output.
# If this had several possible output categories, then the output_dim would be greater,
# and the activation function would be 'softmax', which is a multivariate sigmoid function.
myNeuralNetwork.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# Compiling the ANN (applying stochastic gradient descent)
# For non-binary outcomes, the loss function should be 'categorical_crossentropy'
myNeuralNetwork.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the ANN to the training set
myNeuralNetwork.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# Predicting the Test set results (converted into binary true or false)
# In their natural form, these predictions are measured as probabilities.
y_pred = myNeuralNetwork.predict(X_test)
y_pred = (y_pred > 0.5) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

### EVALUATING THE ARTIFICIAL NEURAL NETWORK        ###

# Applying K-fold cross validation

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Based on how we built the ANN above.
def build_classifier():
    myNeuralNetwork = Sequential()
    myNeuralNetwork.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    myNeuralNetwork.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    myNeuralNetwork.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    myNeuralNetwork.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return myNeuralNetwork

myNeuralNetwork = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)

#n_jobs refers to the number of CPUs used in training. -1 to use all CPUs.
# cv is the number of (c)ross (v)alidation folds to use.
accuracies = cross_val_score(estimator = myNeuralNetwork, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Implementing Dropout