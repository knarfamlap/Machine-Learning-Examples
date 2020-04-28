#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[4]:


# supports both python2 and python3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
 
np.random.seed(42) # make notebook output stable for every run

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[3]:


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


# In[4]:


from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8) # fetch_openml returns targets as strings
sort_by_target(mnist) # fetch_openml returns unsorted dataset


# In[5]:


mnist['data'], mnist['target']


# In[6]:


mnist.data.shape


# In[7]:


X, y = mnist['data'], mnist['target']


# In[8]:


X.shape


# In[9]:


y.shape


# In[10]:


# lets see what a digit looks like
some_digit = X[400]
some_digit_image = some_digit.reshape(28, 28) 
plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation='nearest')

plt.axis('off') 
plt.show() 


# In[11]:


# lets see what the label tells us
y[400]


# In[12]:


def plot_digit(data):
    image  = data.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation='nearest')
    plt.axis('off') 


# In[13]:


# plots all digits
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


# In[14]:


plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()


# In[15]:


# The MNIST dataset is already split into a training set (the first 60,000 images)
# and a test set (the last 10,000 images) 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[16]:


# Lets shuffle the training set to insure that all of our cross-validation folds will be similar
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# # Training A Binary Classifier

# Lets simplify the problem by trying to identify only one digit. 
# Lets choose 5

# In[17]:


# Lets create the target vectors for the classification task
y_train_5 = (y_train == 5) # True for all 5s, false otherwise
y_test_5 = (y_test == 5) 


# In[18]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5) 


# In[19]:


sgd_clf.predict([some_digit])  


# # Preformance

# `cross_val_score()` will allow us to evaluate hour model using K-fold cross-validation.
# K-fold cross validation means splitting the training set into K-folds (in the case below, 3) 
# then making predictions and evaluating them on each fold using a model trained on the remaining folds.

# In[20]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# ## Why accuracy is not a preffered performance measure for classifiers?

# In[21]:


from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool) 


# In[22]:


never_5_clf = Never5Classifier() 
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy") 


# This model has over 90%, this is because only about 10% of the images are 5s, so if you always guess that an image is not a 5, you will be right about 90% of the time. Accuracy is not a good measure of performance for skewed datasets (i.e when some classes are much more frequent than others) 

# # Confusion Matrix

# The general idea is to count the number of times instances of classA are classifier as class B. For example, to know the number of times the classfier confused images of 5s with 3s, you would look in the 5th row and the 3rd column of the confusion matrix

# First we need a set of predicitons, so they can be compared to the actual targets. 
# For this we will user `corss_val_predict()` which is similar to `cross_val_score()`, it preforms K-fold cross validation but instead of returning the evaluation score, it return the predictions made on each test fold. 

# In[23]:


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) 


# In[24]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred) 


# In[25]:


y_train_perfect_predictions = y_train_5


# In[26]:


confusion_matrix(y_train_5, y_train_perfect_predictions)


# A perfect classfier would have only true positives and true negatives so its confusion matrix would have nozero values only on its main diagonal (top left and bottom right) 

# # Precision
# Precision = $\frac{TP}{TP + FP}$
# 
# Precission is used along with anohter metric called $recall$, and also $sensitivity$ or $true positive rate$: this is the ratio of positive instances that are ocrrently detected by the classifier.  
# 
# Recall = $\frac{TP}{TP + FN}$

# In[27]:


from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)


# In[28]:


recall_score(y_train_5, y_train_pred)


# Now the classifier does not look as good as when we used accuracy as a preformance metric. 
# Our classifier claims an image is a 5 88% of the time. Moreover, it only detects 69% of the 5s

# # F1 Score

# Is it is often convinient to combine recall score and preccison intro a single metric called $F_1 score$.
# The $F_1 score$ is the harmonic mean of precision and recall. Where as the regulas mean treats all values equally, the harmonic mean gives much more weight to low values. As a result a classfier will only get a high $F_1 Score$ if both recall and precission are high
# 
# $
# F_1 = 2 * \frac{precision * recall}{precision + recall}
# $

# In[29]:


from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# # ROC Curves
# 
# The ROC (Reciever Operating Characteristic) curve plots the $true positive rate$ (recall) against the $false positive rate$
# 
# The FPR is the ratio of negative inatnces taht are incorrectly classified as postive. FPR is equals to one minus the true negative rate.
# TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus 1 - specificity. 

# In[30]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


# In[31]:


from sklearn.metrics import roc_curve

fpr, tpr, threshholds = roc_curve(y_train_5, y_scores) 


# In[32]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()


# The higher the recall (TPR), the more flase positives (FPR) the classifier produces. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top left corner) 

# # Multiclass Classification

# Scikit-Learn dtects when you try to use abinary classfication a lgorthm for a multiclass classfication task, and it atomatically runs OvA (one versus the rest) 

# In[33]:


sgd_clf.fit(X_train, y_train) 
sgd_clf.predict([some_digit]) 


# Under the hood Scikit-learn actually trained 10 binary classfier, for their decision scores for the image and selected the class with the highes score.
# 
# To see that this is indeed the case, you can call `decision_function()` method. Instead of returning just one score per instance, it now returns 10 scores one per class.

# In[34]:


some_digit_scores = sgd_clf.decision_function([some_digit])


# In[35]:


some_digit_scores


# In[36]:


# The highes score is indeed the one corresponding to class 5
np.argmax(some_digit_scores)


# In[37]:


# When a classifier is trained, it stores the list of target classes in its classes_ attribute
sgd_clf.classes_


# If you want to force Scikit-Learn to use one-versus-one-all, you can us the OnevsOneClassifier or OnevsRestClassifier classes. Simply create an instance and pass a binary classfier to its constructor. 

# In[38]:


from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])


# In[39]:


len(ovo_clf.estimators_) 


# ## Random Forest

# In[40]:


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit]) 


# Random Forest classifiers can directly classify instances into multiple classses. You can call `predict_proba()` to get the list of probabilities that the classifier assigned to each instance for each class

# In[41]:


forest_clf.predict_proba([some_digit]) 


# You can see that the classifier is 100% sure that the given instances is a 0

# ## Evaluating Classifiers

# For evaluating these classifiers, you want to use cross-validation. 

# In[42]:


cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


# In[43]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# Let look at the confusion matrix. You need to make predicitons using the `cross_val_predict()` function, then call the `confusion_matrix()` function

# In[44]:


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# It is often more useful to look at an image representation of the confusion matrix

# In[45]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# # Neural Network Example

# 2-Hidden layers fully connected nearual network (multilayer perceptron)

# In[5]:


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf


# In[ ]:


# parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer
n_hidden_2 = 256 # 2nd layer
num_input = 784 
num_classes = 10 # MNIST total classes

