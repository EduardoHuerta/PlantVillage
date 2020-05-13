from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Add, Activation, Input, concatenate, GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob

# Parametros
batch_size = 64
epochs = 10
input_shape = (224, 224, 3)  #Dimensiones de la iamgen

#direccion donde se localiza el dataset
data_entrenamiento = 'C:/Users/eduar/Escritorio/lalo/Proyecto/RNC/dataset/pv/train/'
data_validacion = 'C:/Users/eduar/Escritorio/lalo/Proyecto/RNC/dataset/pv/val/'

mobnet = MobileNetV2(weights='imagenet', include_top=False,input_shape=input_shape) # Hasta ultima capa CNN, entrada será de CIFAR10

# Opciones para iniciar modelos previamente entrenados
# Selección de salida, ultima capa CNN
# Añadimos capas a modelo a entrenar
x = mobnet.output # tomamos salida de MobilenetV2 (modelo base)

x = Flatten()(x)
x = Dense(128,activation='relu')(x)
predicciones = Dense(38, activation='softmax')(x) # Determinamos salida a 38 clases


##CREAR LA RED MobileNet

# Modelo a entrenar
new_model = Model(inputs=mobnet.input, outputs=predicciones)
# Resumen del nuevo modelo (El tamaño mínimo de imagen para VGG16 es 32x32)
#new_model.summary()

    
# Si queremos congelar solo las 10 primeras y reentrenar el resto (pesos iniciales son de ImageNet)
for capa in mobnet.layers[:135]: # Se cuentan de arriba a abajo en summary, incluyendo Input y pooling
   capa.trainable = False
for capa in mobnet.layers[135:]: # Entrenamiento será a ultima CNN de bloque 5 y FC
   capa.trainable = True

new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
new_model.summary()

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

datagen = ImageDataGenerator(rescale=1./255)  

# Cargar e iterar en training dataset
x_train = entrenamiento_datagen.flow_from_directory(
                        data_entrenamiento
                      , target_size=(224, 224)
                      , batch_size=64
                      , class_mode='categorical')
# Cargar e iterar test dataset
x_test = datagen.flow_from_directory(
                        data_validacion,
                        target_size=(224, 224), 
                        batch_size=64,
                        class_mode='categorical')

historia = new_model.fit_generator(x_train,
                                   validation_data=x_test,
                                   epochs=epochs,
                                   steps_per_epoch=len(x_train),
                                   validation_steps= len(x_test),
                                   verbose=1)

score = new_model.evaluate(x_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:',score[1])

new_model.save('MobileNetV2_PlantVillage.h5')
new_model.save_weights('MobileNetV2_PlantVillage_pesos.h5')