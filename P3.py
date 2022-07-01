# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:46:35 2022

@author: jases

You are tasked to perform image classification to classify concretes with cracks

Link to dataset:
    https://data.mendeley.com/datasets/5y9wdsg2zt/2

You may apply transfer learning.
"""
#1. Import packages
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers
import datetime
import pathlib
import cv2

#2. Read the data
#_URL = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip'
#path_to_zip = keras.utils.get_file('concretes_with_cracks.zip',origin=_URL,extract=True)
#file_path = os.path.join(os.path.dirname(path_to_zip),'concretes_with_cracks')

#data_dir=patoolib.extract_archive("Concrete Crack Images for Classification.rar", outdir="C:/Users/jases/.keras/datasets")
file_path= r"C:\Users\jases\Desktop\AI-06\DL\data\Concrete Crack Images for Classification"
data_dir= pathlib.Path(file_path)

image_count= len(list(data_dir.glob('*/*.jpg')))
print(image_count)

#3. Data preprocessing
SEED=12345
BATCH_SIZE=10
IMG_SIZE= (100, 100)

train_dataset= keras.utils.image_dataset_from_directory(data_dir,validation_split= 0.3,subset='training',seed= SEED,shuffle= True,image_size=IMG_SIZE,batch_size= BATCH_SIZE)


validation_dataset= keras.utils.image_dataset_from_directory(data_dir,validation_split= 0.3,subset='validation',seed= SEED,shuffle= True,image_size=IMG_SIZE,batch_size= BATCH_SIZE)

class_names= train_dataset.class_names
print(class_names)

plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax= plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

AUTOTUNE= tf.data.AUTOTUNE

train_dataset= train_dataset.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
validation_dataset= validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

#%%
#3. Create data augmentation pipeline
data_augmentation= keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

for images, labels in train_dataset.take(1):
    first_image= images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image= data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

#Create a layer for data preprocessing
preprocess_input= applications.mobilenet_v2.preprocess_input

#Create the base model by using MobileNetV2
IMG_SHAPE= IMG_SIZE + (3,)
base_model= applications.MobileNetV2(input_shape= IMG_SHAPE, include_top= False, weights='imagenet')

#Apply layer freezing
for layer in base_model.layers[:1000]:
    layer.trainable= False
    
base_model.summary()

nClass= len(class_names)

global_avg_pooling= layers.GlobalAveragePooling2D()
output_layer= layers.Dense(nClass, activation='softmax')

#%%
#Use Functional API to construct model
inputs= keras.Input(shape= IMG_SHAPE)
x= data_augmentation(inputs)
x= preprocess_input(x)
x= base_model(x)
x= global_avg_pooling(x)
x= keras.layers.Dropout(0.3)(x)
outputs= output_layer(x)

model= keras.Model(inputs= inputs, outputs= outputs)
model.summary()

#Compile the model
optimizer= keras.optimizers.Adam(learning_rate= 0.0001)
loss= keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer= optimizer, loss= loss, metrics=['accuracy'])

#Train the model
EPOCHS=1

base_log_path = r"C:\Users\jases\Desktop\AI-06\DL\tb_logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ '___Project3')
tb = keras.callbacks.TensorBoard(log_dir=log_path)

history = model.fit(train_dataset,validation_data=validation_dataset, epochs=EPOCHS,callbacks=[tb])

#%%
#Make prediction from image given
#Display image
file_prediction_path= r"C:\Users\jases\Desktop\AI-06\DL\prediction\concrete.jpg"
predict_concrete= cv2.imread(file_prediction_path)
predict_concrete=  cv2.resize(predict_concrete, (256, 256))
cv2.imshow("Predict Concrete", predict_concrete)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img= tf.keras.utils.load_img(file_prediction_path,target_size= IMG_SIZE)

img_array= tf.keras.utils.img_to_array(img)
img_array= tf.expand_dims(img_array, 0) 

predictions= model.predict(img_array)
score= tf.nn.softmax(predictions[0])

print("This image most likely belongs to {} with a {:2f} percent confidence". format(class_names[np.argmax(score)], 100*np.max(score)))

















