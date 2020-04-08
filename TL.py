from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model,Sequential
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

Imagenes_dimensiones = [224, 224]

data_entrenamiento = 'dataset/pv/train'
data_validacion = 'dataset/pv/val'

vgg = VGG16(input_shape= Imagenes_dimensiones + [3], weights= 'imagenet', include_top= False)

for capa in vgg.layers:
    capa.trainable = False

folders = glob('dataset/pv/train/*')

x = Flatten()(vgg.output)

prediccion = Dense(len(folders), activation='softmax')(x)


##CREAR LA RED VGG16

model = Model(inputs=vgg.inputs, outputs=prediccion)

model.summary()

model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
            )

from keras.preprocessing.image import ImageDataGenerator

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size= (224,224),
    batch_size= 32,
    class_mode='categorical')


validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(224, 224),
    batch_size=32 ,
    class_mode='categorical')

historia = model.fit_generator(
    entrenamiento_generador,
    validation_data=validacion_generador,
    epochs=5,
    steps_per_epoch=len(entrenamiento_generador),
    validation_steps= len(validacion_generador)
    )


#loss
plt.plot(historia.history['loss'], label= 'training loss')
plt.plot(historia.history['val_loss'], label= 'val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#accuracies
plt.plot(historia.history['acc'], label= 'training accuracy')
plt.plot(historia.history['val_acc'], label= 'val accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
 
import tensorflow as tf
from keras.models import load_model

model.save('ModeloPlantVillage.h5')
model.save_weights('PlantVillage_pesos.h5')