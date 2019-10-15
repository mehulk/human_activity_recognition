# -*- coding: utf-8 -*-


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

os.chdir(r'C:\Users\Hp\Desktop\Documents\IITR-CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\LBP_feat_images');

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
