from keras.applications.xception import Xception, preprocess_input
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

# Parametros
batch_size = 128
num_classes = 38
epochs = 5
input_shape = (299, 299, 3)  #Dimensiones de la iamgen

#direccion donde se localiza el dataset
data_entrenamiento = 'H:/Data/training'
data_validacion = 'H:/Data/test'

xcep = Xception(weights='imagenet', include_top=False,input_shape=input_shape) # Hasta ultima capa CNN, entrada será de CIFAR10
xcep.summary()


# Etiquetamos como NO entrenables las capas CNN, también podemos seleccionar a mano cuales son entrenables
for capa in xcep.layers:
    capa.trainable = False

folders = glob('H:/Data/training/*')

# Opciones para iniciar modelos previamente entrenados
# Selección de salida, ultima capa CNN
# Añadimos capas a modelo a entrenar
x = xcep.output # tomamos salida de MobilenetV2 (modelo base)

x = GlobalAveragePooling2D()(x)
predicciones = Dense(len(folders), activation='softmax')(x) # Determinamos salida a 38 clases


##CREAR LA RED Xception

#predicciones = Dense(num_classes, activation='softmax')(x) # Determinamos salida a 38 clases

# Modelo a entrenar
new_model = Model(inputs=xcep.input, outputs=predicciones)
# Resumen del nuevo modelo (El tamaño mínimo de imagen para VGG16 es 32x32)
new_model.summary()

    
# Si queremos congelar solo las 10 primeras y reentrenar el resto (pesos iniciales son de ImageNet)
for capa in xcep.layers[:100]: # Se cuentan de arriba a abajo en summary, incluyendo Input y pooling
   capa.trainable = False
for capa in xcep.layers[100:]: # Entrenamiento será a ultima CNN de bloque 5 y FC
   capa.trainable = True

new_model.compile(optimizer='adam'
                , loss='categorical_crossentropy'
                , metrics=['accuracy'])
new_model.summary()

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

datagen = ImageDataGenerator(rescale=1./255)  

# Cargar e iterar en training dataset
x_train = entrenamiento_datagen.flow_from_directory(
                        data_entrenamiento
                      , target_size=(299, 299)
                      , batch_size=64
                      , class_mode='categorical')
# Cargar e iterar test dataset
x_test = datagen.flow_from_directory(
                        data_validacion,
                        target_size=(299, 299), 
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
print('Test accuracy:', score[1])

#loss
plt.plot(historia.history['loss'], label= 'training loss')
plt.plot(historia.history['val_loss'], label= 'val loss')
plt.legend()
plt.show()
plt.savefig('Xception_LossVal_loss')

#accuracies
plt.plot(historia.history['accuracy'], label= 'training accuracy')
plt.plot(historia.history['val_accuracy'], label= 'val accuracy')
plt.legend()
plt.show()
plt.savefig('Xception_AccVal_acc')


new_model.save('Xception_PlantVillage.h5')
new_model.save_weights('Xception_PlantVillage_pesos.h5')