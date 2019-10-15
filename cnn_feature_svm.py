# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:51:24 2019

@author: Hp
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Hp\Desktop\python\CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\cnn_feat');

'''Loading Data'''
X_train = np.load('X_train.npy');
X_test  = np.load('X_test.npy');
y_train = np.load('y_train.npy');
y_test  = np.load('y_test.npy');

#%%

'''Preprocessing'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train);
X_test = sc_X.transform(X_test);

'''Classifier '''
from sklearn.svm import SVC
classifier = SVC(C=100,kernel='rbf');
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
y_pred_test = classifier.predict(X_test);
print("\nFor test data:\t");
print(classification_report(y_test,y_pred_test));
print(accuracy_score(y_test,y_pred_test));
cm_test = confusion_matrix(y_test,y_pred_test);
plt.matshow(cm_test);

