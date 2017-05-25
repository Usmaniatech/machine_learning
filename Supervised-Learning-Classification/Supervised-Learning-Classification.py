
# coding: utf-8

# In[1]:

# Supervised Learning with Scikit Learn


# In[2]:

# Supervised learning uses labeled data. It uses predictor/features/explanatory variables & target/response variable.


# In[3]:

# There are 2 techniques in supervised learning: 1. Classification 2. Regression


# In[4]:

# If the target variable is continuous, like you want to predict the house prices based on its features, its a Regression problem.


# In[5]:

# If the target variable is categorical, 
# like you want to predict if the google shares will increase next month or not, or employee will leave the company or not, 
# its a Classification problem.


# In[9]:

# We will be focusing on Classification, in this notebook


# In[24]:

# import dataset
from sklearn import datasets


# In[25]:

import pandas as pd


# In[26]:

import numpy as np


# In[27]:

import matplotlib.pyplot as plt


# In[61]:

# Load the digits dataset: digits
digits = datasets.load_digits()


# In[62]:

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)


# In[63]:

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)


# In[64]:

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# In[65]:

#  It looks like the image corresponds to the digit '5'. 
# Now, can you build a classifier that can make this prediction not only for this image,
# but for all the other ones in the dataset


# In[66]:

# After creating arrays for the features and target variable, 
# you will split them into training and test sets, 
# fit a k-NN classifier to the training data, 
# and then compute its accuracy using the .score() method.


# In[69]:

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[70]:

# Create feature and target arrays
X = digits.data
y = digits.target


# In[71]:

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# for details
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


# In[72]:

# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


# In[73]:

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)


# In[74]:

# Fit the classifier to the training data
knn.fit(X_train, y_train)


# In[75]:

# Print the accuracy
print(knn.score(X_test, y_test))


# In[76]:

# This out of the box k-NN classifier with 7 neighbors has learned from the training data and predicted the labels of the images in the test set with 98% accuracy, 
# and it did so in less than a second! This is one illustration of how incredibly useful machine learning techniques can be.


# In[77]:

# Model Complexity
# Larger value of K = smoother desicion boundry = less complex model
# Smaller K = more complex model = can lead to overfitting

# If you increase K even more, model will become less well on both traing and test set
# and it generate underfitting


# In[78]:

# Now you will compute and plot the training and testing accuracy scores for a variety of different neighbor values. 
# By observing how the accuracy scores differ for the training and testing sets with different values of k, 
# you will develop your intuition for overfitting and underfitting.


# In[79]:

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# In[80]:

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)


# In[81]:

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[82]:

# test accuracy is highest when using 3 and 5 neighbors. 
# Using 8 neighbors or more seems to result in a simple model that underfits the data.


# In[ ]:



