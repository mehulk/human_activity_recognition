# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:35:34 2019

@author: Hp
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

os.chdir(r'C:\Users\Hp\Desktop\python\CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\cnn_feat');

'''Loading Data'''
X_train_cnn = np.load('X_train.npy');
X_test_cnn  = np.load('X_test.npy');
y_train_cnn = np.load('y_train.npy');
y_test_cnn  = np.load('y_test.npy');

#%%

'''Preprocessing'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train_cnn = sc_X.fit_transform(X_train_cnn);
X_test_cnn = sc_X.transform(X_test_cnn);

'''classifier_cnn '''
from sklearn.svm import SVC
classifier_cnn = SVC(C=100, kernel='rbf', probability=True);
classifier_cnn.fit(X_train_cnn,y_train_cnn)
y_pred_train = classifier_cnn.predict(X_train_cnn);

#%%

'''For Training Data '''
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
print("\nFor training data:\t");
print(classification_report(y_train_cnn,y_pred_train));
print(accuracy_score(y_train_cnn,y_pred_train));
cm_train = confusion_matrix(y_train_cnn,y_pred_train);
plt.matshow(cm_train)

#%%

''' For Testing Data'''
y_pred_test_cnn = classifier_cnn.predict_proba(X_test_cnn);
print("\nFor test data:\t");


#%%

'''LBP_AUG'''

os.chdir(r'C:\Users\Hp\Desktop\python\CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\LBP_feat_images');

'''Loading Data'''
matX_train = scipy.io.loadmat('X_train.mat');
matY_train = scipy.io.loadmat('y_train.mat');
matX_test  = scipy.io.loadmat('X_test.mat');
matY_test  = scipy.io.loadmat('y_test.mat');

X_train = matX_train.get('X_train');
X_test  = matX_test.get('X_test');
y_train = matY_train.get('y_train');
y_test  = matY_test.get('y_test');

#%%

'''Preprocessing'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train);
X_test = sc_X.transform(X_test);

'''Classifier '''
from sklearn.svm import SVC
classifier = SVC(C=100,kernel='rbf',probability=True);
classifier.fit(X_train,y_train)
y_pred_train = classifier.predict(X_train);

#%%

'''For Training Data '''
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
print("\nFor training data:\t");
print(classification_report(y_train,y_pred_train));
print(accuracy_score(y_train,y_pred_train));
cm_train = confusion_matrix(y_train,y_pred_train);
plt.matshow(cm_train)

#%%

''' For Testing Data'''
y_pred_test = classifier.predict_proba(X_test);
print("\nFor test data:\t");

y_pred = np.add(0.2*y_pred_test,0.8*y_pred_test_cnn);

y_pred = np.argmax(y_pred, axis=1 )
y_pred=y_pred+1;
print(accuracy_score(y_test, y_pred))

cm_train = confusion_matrix(y_test,y_pred);
plt.matshow(cm_train)

