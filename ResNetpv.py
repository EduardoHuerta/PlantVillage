from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Add, Activation, Input, concatenate, GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix

# Parametros
batch_size = 64
epochs = 10
input_shape = (224, 224, 3)  #Dimensiones de la iamgen

#direccion donde se localiza el dataset
data_entrenamiento = 'H:/Data/training'
data_validacion = 'H:/Data/test'

resnet = ResNet50V2(weights='imagenet', include_top=False,input_shape=input_shape) # Hasta ultima capa CNN, entrada será de CIFAR10
resnet.summary()

for capa in resnet.layers:
    capa.trainable = False

folders = glob('H:/Data/training/*')

# Opciones para iniciar modelos previamente entrenados
# Selección de salida, ultima capa CNN
# Añadimos capas a modelo a entrenar
x=Dropout(0.5)(resnet.output)
x = Flatten()(x) # tomamos salida de Resnet50 (modelo base)

predicciones = Dense(len(folders), activation='softmax')(x) # Determinamos salida a 38 clases


##CREAR LA RED ResNet

# Modelo a entrenar
new_model = Model(inputs=resnet.input, outputs=predicciones)
# Resumen del nuevo modelo
new_model.summary()


new_model.compile(optimizer='adam'
                , loss='categorical_crossentropy'
                , metrics=['accuracy'])
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
                      , batch_size=32
                      , class_mode='categorical')
# Cargar e iterar test dataset
x_test = datagen.flow_from_directory(
                        data_validacion,
                        target_size=(224, 224), 
                        batch_size=32,
                        class_mode='categorical')

historia = new_model.fit_generator(x_train,epochs=epochs, verbose=1, validation_data=x_test)

x_test = datagen.flow_from_directory('H:/Data/test/',target_size=(224, 224), batch_size=32, shuffle=False)
score = new_model.evaluate(x_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

new_model.save('ResNet50V2_PlantVillage.h5')

y_pred = new_model.predict_generator(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = x_test.labels
target_names=list(x_test.class_indices.keys())
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=target_names))