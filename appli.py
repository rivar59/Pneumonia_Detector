# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
  
# open method used to open different extension image file

num_classes = 2
model2 = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(128,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16,4, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(num_classes, activation='softmax')
        ])

model2.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],)

model2.load_weights('./checkpoints/my_checkpoint')
    


st.title("Pneumonia detector")
def neural_network(image):
    im = Image.open(image) 
    image_to_predict = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    img_to_predict = np.expand_dims(cv2.resize(image_to_predict,(200,200)), axis=0) 
    predict_x=model2.predict(img_to_predict)
    classes_x=np.argmax(predict_x,axis=1)
    print(classes_x[0])
    if classes_x[0] == 0:
        st.header("This is normal")
    elif classes_x[0] == 1 :
        st.header("He is sick")
        
        
        
file_object = st.file_uploader('upload your X-ray chest')
if file_object != None:
    neural_network(file_object)
