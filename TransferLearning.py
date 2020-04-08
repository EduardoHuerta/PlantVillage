import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications

vgg=applications.vgg16.VGG16() #Variable que tendra el modelo con todas las capas necesarias

vgg.summary() #Para observar todo el modelo

cnn=Sequential() #Modelo secuencial

for capa in vgg.layers:
    cnn.add(capa) #Queremos que añadas esa capa a nuestra variable de cnn

cnn.summary() #Observamos la misma estructura que vgg pero cnn es nuestro modelo

cnn.pop() #Al ejecutar nos desicimos de la capa de prediccion se a eliminado

cnn.summary() #Nos quedamos con todas las capas ya entrenadas sin necesidad de reentrenarlos

for layer in cnn.layers:  #Por cada cada en la estructura
    layer.trainable=False #Durante todo el entrenamiento para clasificacion no modifique ningun peso 

#38 es el numero de clases del dataset en total
cnn.add(Dense(38,activation='softmax')) 

cnn.summary()#Nos muestra 38 clases en lugar de 1000 al principio

def modelo():  #Funcion 
    vgg=applications.vgg16.VGG16()
    cnn=Sequential()
    for capa in vgg.layers: #Cpiamos todas las capas del modelo vgg al modelo cnn
        cnn.add(capa)
    cnn.layers.pop()   #Eliminamos la ultima capa que clasifica mil elementos
    for layer in cnn.layers:  
        layer.trainable=False  #Solo entrene la ultima capa que se a añadido que tiene la funcion de softmax
    cnn.add(Dense(3,activation='softmax')) 
    
    return cnn  #Regresamos el modelo listo para ser entrenado

K.clear_session()



data_entrenamiento = 'dataset/pv/train'
data_validacion = 'dataset/pv/val'


epocas=20
longitud, altura = 224, 224
batch_size = 32
pasos = 1000
validation_steps = 300
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 38
lr = 0.004


##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

##TODO ESTO ES SUSTITUIDO POR LA FUNCION QUE CREA LA RED VGG16
'''cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))
'''


##CREAR LA RED VGG16

cnn=modelo()

#cnn.compile(loss='categorical_crossentropy',
#            optimizer='adam',
#            metrics=['accuracy'])


cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')    