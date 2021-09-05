import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cv2
cwd = os.getcwd()

def Seperate(loadfile,savefolder):   
    img = cv2.imread(loadfile + ".jpg",0)
    img = 255 - img
    ydone = False
    start_ypoints_crop = []
    end_ypoints_crop = []
    for row in range(len(img)):
        if any([ind < 200 for ind in iter(img[row])]) and not ydone:
            start_ypoints_crop.append(row)
            ydone = True
        if all([ind > 200 for ind in iter(img[row])]) and ydone:
            end_ypoints_crop.append(row)
            ydone = False

    CroppedImages= []
    for top,bot in zip(start_ypoints_crop,end_ypoints_crop):
        CroppedImages.append(img[top:bot,...])

    xdone = False
    start_xpoints_crop = []
    end_xpoints_crop = []
    for k in range(len(CroppedImages)):
        start_xpoints_crop.append([])
        end_xpoints_crop.append([])
    for i in range(len(CroppedImages)):
        for col in range(len(CroppedImages[0][0])):
            if any([ind < 200 for ind in iter(CroppedImages[i][...,col])]) and not xdone:
                start_xpoints_crop [i].append(col)
                xdone = True
            if all([ind > 200 for ind in iter(CroppedImages[i][...,col])]) and xdone:
                end_xpoints_crop [i].append(col)
                xdone = False

    Numbers=[]
    for i in range(len(CroppedImages)):
        for left,right in zip(start_xpoints_crop[i],end_xpoints_crop[i]):
            Numbers.append(CroppedImages[i][...,left-2:right+2])

    for i in range(len(Numbers)):
        cv2.copyMakeBorder(Numbers[i],4,4,10,10,cv2.BORDER_CONSTANT,value=255)
        cv2.imwrite(f'Images/{savefolder}/{i}.png',Numbers[i])
        
folders = ['Zeros','Ones','Twos','Threes','Fours','Fives','Sixes','Sevens','Eights','Nines']
final_images = []
for n in range(len(folders)):   
    for file in glob.glob(f"{cwd}RawImages/{folders[n]}/*.jpg"):
        image = cv2.imread(file,0)
        image = cv2.copyMakeBorder(image,2,2,10,10,cv2.BORDER_CONSTANT,value=255)
        image = cv2.resize(image,(28,28))                
        image = image / np.max(image)
        final_images.append(image)
        
from keras.utils import to_categorical

labels = []
final_labels = []
for n in range(len(folders)):   
    for file in glob.glob(f"{cwd}{folders[n]}/*.jpg"):
        labels.append(n)
final_labels = to_categorical(labels,num_classes = 10)

from sklearn.model_selection import train_test_split
img_train, img_test, lbl_train, lbl_test = train_test_split(final_images, final_labels,
                                                            test_size=0.2, random_state=42, stratify=final_labels)
                                                           
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

CNT_Model = Sequential()


CNT_Model.add(Conv2D(20,5, activation = 'relu', padding = 'same',kernel_regularizer = None, input_shape =(28,28,1)))
CNT_Model.add(MaxPooling2D(pool_size = (2,2)))
CNT_Model.add(Conv2D(20,5, activation = 'relu', padding = 'same', kernel_regularizer = None))
CNT_Model.add(MaxPooling2D(pool_size = (2,2)))
#CNT_Model.add(Dropout(0.5))
CNT_Model.add(Flatten())
CNT_Model.add(Dropout(0.5))
#CNT_Model.add(BatchNormalization())
CNT_Model.add(Dense(64,activation = 'relu', kernel_regularizer = None))
#CNT_Model.add(BatchNormalization())
CNT_Model.add(Dense(10,activation='sigmoid'))

Optimizer = keras.optimizers.SGD(lr=0.01, momentum = 0.5, decay = 1e-5, nesterov = True)

CNT_Model.compile(optimizer=Optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
CNT_Model.summary()


img_train = np.reshape(img_train,(len(img_train),28,28,1))

from keras.callbacks import ModelCheckpoint
ann_hist = CNT_Model.fit(np.array(img_train), np.array(lbl_train), validation_split = 0.1, batch_size = 10, epochs = 2)


img_test = np.reshape(img_test,(len(img_test),28,28,1))

test_loss, test_acc = CNT_Model.evaluate(np.array(img_test), np.array(lbl_test), batch_size = 3)
CNT_Model.save('Model1.h5')
labels_predicted = CNT_Model.predict(np.array(img_test))
lbl= np.argmax(labels_predicted, axis=1)
print('Test loss is',test_loss)
print('Test accuracy is',test_acc)