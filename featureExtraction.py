# -*- coding: utf-8 -*-

import os
import numpy as np
import keras
from keras.preprocessing import image
from keras import Model
from keras.models import load_model

#%%
#
os.chdir(r'C:\Users\Hp\Desktop\Documents\IITR-CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\S')

model = load_model('activity_recog_top_d2.h5')
'''Front - activity_recog_front.h5
    Side - activity_recog_side_d.h5
    top  -activity_recog_top_d2.h5
    '''


#%%

model.summary();

#%%

layer_name = 'fc-2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_layer_model.summary()

#%%

def prepare_image(file):
    img = image.load_img(file,target_size=(224,224));
    img_array = image.img_to_array(img);
    img_array_expanded_dims = np.expand_dims(img_array,axis=0);
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

#%%
    
X_train = np.zeros((292,1024));
y_train = np.zeros((292))

i=0;
for activity in range(1,21):
    img_path='train/%d/' %(activity);    
    for filename in os.listdir(img_path):
        if filename.endswith(".jpg"):
            img = prepare_image(img_path+filename);
            feat = intermediate_layer_model.predict(img);
            X_train[i]=feat;
            y_train[i]=activity;
            print(i);
            i = i+1;
            continue
        else:
            continue                              
           
   
#%%

np.save('X_train_top',X_train);
np.save('y_train_top',y_train);

#%%

X_test = np.zeros((3504,1024));
y_test = np.zeros((3504))

i=0;
for activity in range(1,21):
    img_path='test/%d/' %(activity);    
    for filename in os.listdir(img_path):
        if filename.endswith(".jpg"):
            img = prepare_image(img_path+filename);
            feat = intermediate_layer_model.predict(img);
            X_test[i]=feat;
            y_test[i]=activity;
            print(i);
            i = i+1;
            continue
        else:
            continue 
        
#%%

np.save('X_test_top',X_test);
np.save('y_test_top',y_test);         
        

