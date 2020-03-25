'''import os


def root_path():
    return os.path.dirname(__file__)


def dataset_path():
    return os.path.join(root_path(),"dataset")


def src_path():
    return os.path.join(root_path(),"src")

def output_path():
    return os.path.join(root_path(),"output")

def weight_path():
    return os.path.join(root_path(),"weight")'''
    # Capas
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense

# Flattening(Aplanamiento), regularization, inter-layer data manipulation
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam

# Pre-procesamiento
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


# PARTE 2 - Encajando RNC en las imagenes

# Model optimizer
optimizer = Adam(lr=0.0001)
# Dimensiones de Imagen dimms;
img_rows, img_cols = 256, 256
input_shape = (img_rows, img_cols, 3)
# Parametros de entrenamiento
batch_size = 60
epochs = 10
seed= 69


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="dataset/PlantVillage", #Direccion donde se localiza el dataset
    class_mode="categorical",
    seed=seed
)

test_generator = test_datagen.flow_from_directory("dataset/PlantVillage")

print("input shape", input_shape)

# CREANDO EL MODELO

# Inicializando RNC

model = Sequential()
# Bloque 1
# Parte 1 - Convolucion
model.add(Conv2D(input_shape=input_shape ,filters=16, kernel_size=(3, 3), strides=3, activation='relu'))
# Parte 2 - Agrupacion
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Bloque 2 Capa Extra - Convolucion
# Parte 1 - Convolucion
model.add(Conv2D(input_shape=input_shape ,filters=32, kernel_size=(3, 3), strides=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Flattening, but better: Instead of directly flattening,
# global max pooling first pools all feature maps together,
# then chugs into an FC layer
# Parte 3 - Aplanamiento
model.add(Flatten())

# Decision layer
# Parte 4 - Conexion Completa
model.add(Dense(38, activation='softmax'))
model.summary()

# Compilacion y entrenamiento
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=1,
    epochs=epochs)

model.evaluate_generator(generator=test_generator, steps=50)

#salvando el modelo
filepath="C:/Users/eduar/Escritorio/lalo/Proyecto/model.hdf5"
model.save(filepath)
filepath2="C:/Users/eduar/Escritorio/lalo/Proyecto/pesos.h5"
model.save_weights(filepath2)

#Graficando los valores de entrenamiento
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['acc']
val_acc = history.history['acc']
loss = history.history['loss']
val_loss = history.history['loss']
epochs = range(1, len(loss) + 1)

#Graficacion de la precision
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Precision del Training y Validacion')
plt.ylabel('Accuracy')
plt.xlabel('Epocas')
plt.legend(['Train', 'Test'], loc='upper left')

#Graficacion de la perdida
plt.plot(epochs, loss, color='orange', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Perdida en el Training y Validation')
plt.xlabel('Epocas')
plt.ylabel('Loss')
plt.legend()

plt.show()
