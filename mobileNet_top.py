# -*- coding: utf-8 -*-


"""TOP"""
import os
import matplotlib.pyplot as plt
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import *  
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
import math

os.chdir(r'C:\Users\Hp\Desktop\Documents\IITR-CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\T')
mobile = keras.applications.mobilenet.MobileNet();

#%%

train_path = 'train/';
test_path = 'test/';
train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path,
                                  target_size=(224,224),batch_size=36)
                                   
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path,
                                 target_size=(224,224),batch_size=275,shuffle=False)        
                           
#%%

x = mobile.layers[-6].output;
x = Dense(1024, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
predictions = Dense(20,activation='softmax')(x);
model_top = Model(inputs = mobile.input,output = predictions);
model_top.compile(Adam(lr=.0001),loss = 'categorical_crossentropy',metrics=['accuracy']);

#%%

def step_decay(epoch):
   initial_lrate = 0.0001
   drop = 0.5
   epochs_drop = 30.0
   lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))   
   return lrate

lrate = LearningRateScheduler(step_decay)

model_top.fit_generator(train_batches,steps_per_epoch=5 ,epochs=120, callbacks=[lrate])

 #%%                     
       
test_labels = test_batches.classes;
test_batches.class_indices;
prediction = model_top.predict_generator(test_batches,steps =1);
prediction = prediction.argmax(axis =1);

#%%

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(test_labels,prediction));
print(classification_report(test_labels,prediction));
cm_train = confusion_matrix(test_labels,prediction);
plt.matshow(cm_train);

#%%
from keras.models import load_model
model_top = load_model('activity_recog_top_d.h5')
#%%
model_top.save('activity_recog_top_d2.h5');