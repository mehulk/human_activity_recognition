# -*- coding: utf-8 -*-


import os
import os.path
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#
os.chdir(r'C:\Users\Hp\Desktop\Documents\IITR-CV\DMM_CRC\DMM_CRC\MSR-Action3D\MobileNet\S')


#%%
total=0;
for activity in range(1,21):    
    for subject in [1 , 3 , 5 , 7 ,9]:
        for task in range(1,4):
            
            if activity < 10:   #
                img_path='train/%d/sa0%d_s0%d_e0%d_sdepth.jpg' %(activity,activity,subject,task);
            else:           #
                img_path='train/%d/sa%d_s0%d_e0%d_sdepth.jpg' %(activity,activity,subject,task);                              
            
            if os.path.exists(img_path):                
                img= image.load_img(img_path)
                img = image.img_to_array(img)
                img = img/224;
                plt.imshow(img);
                
                gen = ImageDataGenerator(
                    rotation_range = 15,
                    shear_range=15,
                    height_shift_range=5,
                    #horizontal_flip=True,
                    fill_mode='constant'
                   )
            
                input_image=img.reshape(1,*img.shape)
                gen.fit(input_image)
                
                i=0;
                save_address ='train/%d' %(activity);
                for batch in gen.flow(input_image,batch_size=1,
                                      save_to_dir=save_address,
                                      save_prefix='aug_%d_%d_0' %(subject,task),
                                      save_format='jpg'):
                    
                    i+=1;
                    if i==11:
                        total+=i;
                        break;
            else:
                print(img_path);
            
print(total);
                        
    
#%%        
                          
             
img_path='train/10/ta10_s05_e02_sdepth.jpg';
img= image.load_img(img_path)
img = image.img_to_array(img)
img = img/224;
plt.imshow(img);

gen = ImageDataGenerator(
        rotation_range = 15,
        shear_range=15,
        height_shift_range=5,
        #horizontal_flip=True,
        fill_mode='constant'
        )


input_image=img.reshape(1,*img.shape)
gen.fit(input_image)
i=0;
for batch in gen.flow(input_image,batch_size=1,
                     save_to_dir='train/10',save_prefix='aug_5_2_1',save_format='jpg'):
    plt.figure();
    imgplot = plt.imshow(image.img_to_array(batch[0]));
    i+=1;
    if i==1:
        break;
    plt.axis('off');
    plt.show();


