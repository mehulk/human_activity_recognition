# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Hp\Desktop\Documents\IITR-CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\F');

X_train_f = np.load('X_train_front.npy');
X_test_f  = np.load('X_test_front.npy');
y_train_f = np.load('y_train_front.npy');
y_test_f  = np.load('y_test_front.npy');

#%%

os.chdir(r'C:\Users\Hp\Desktop\python\CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\S');

X_train_s = np.load('X_train_side.npy');
X_test_s  = np.load('X_test_side.npy');
y_train_s = np.load('y_train_side.npy');
y_test_s  = np.load('y_test_side.npy');

#%%

os.chdir(r'C:\Users\Hp\Desktop\python\CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\T');

X_train_t = np.load('X_train_top.npy');
X_test_t  = np.load('X_test_top.npy');
y_train_t = np.load('y_train_top.npy');
y_test_t  = np.load('y_test_top.npy');

#%%

os.chdir(r'C:\Users\Hp\Desktop\python\CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet');

X_train = np.append(X_train_f, X_train_s, axis =1);
X_test  = np.append(X_test_f, X_test_s, axis =1);

X_train = np.append(X_train, X_train_t, axis =1);
X_test  = np.append(X_test, X_test_t, axis =1);

#%%

if(np.array_equal(y_train_f , y_train_s) and np.array_equal(y_train_f , y_train_t)):
    y_train = y_train_f;
    print("Equal")
else:
    print("ERROR")
    
if(np.array_equal(y_test_f , y_test_s) and np.array_equal(y_test_f , y_test_t)):
    y_test = y_test_f;
    print("Equal")
else:
    print("ERROR")    
   
#%%    

np.save('X_train',X_train);
np.save('y_train',y_train);

np.save('X_test',X_test);
np.save('y_test',y_test);    

        

