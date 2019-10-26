# -*- coding: utf-8 -*-
"""
Introduction file for Udemy's Deep Learning course.

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
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

