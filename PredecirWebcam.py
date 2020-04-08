import pickle
import numpy as np
import cv2
import tensorflow as tf

###########################################

Ancho_frame = 360  #Resolucion de la camara
Longitud_frame = 240
brillo = 180 
threshold = 0.9
font = cv2.FONT_HERSHEY_COMPLEX

###########################################

#Poner a funcionar la video camara
captura = cv2.VideoCapture(0)
captura.set(3,Ancho_frame)
captura.set(4,Longitud_frame)
captura.set(10, brillo)

#importamos el modelo entrenado
model1 = tf.keras.models.load_model('ModeloPlantVillage.h5')
#model1 = tf.keras.models.load_model('model.hdf5')

def preprocesamiento(img):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    img = img/255
    return img

def getNombreClases(ClasesNo):
    if ClasesNo == 0:    return 'Costra de Manzana'
    elif ClasesNo == 1:  return 'apple black rot'
    elif ClasesNo == 2:  return 'Cedar_apple_rust'
    elif ClasesNo == 3:  return 'Apple___healthy'
    elif ClasesNo == 4:  return 'Blueberry___healthy'
    elif ClasesNo == 5:  return 'Cherry_(including_sour)___healthy'
    elif ClasesNo == 6:  return 'Cherry_(including_sour)___Powdery_mildew'
    elif ClasesNo == 7:  return 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'
    elif ClasesNo == 8:  return 'Corn_(maize)___Common_rust'
    elif ClasesNo == 9:  return 'Corn_(maize)___healthy'
    elif ClasesNo == 10: return 'Corn_(maize)___Northern_Leaf_Blight'
    elif ClasesNo == 11: return 'Grape___Black_rot'
    elif ClasesNo == 12: return 'Grape___Esca_(Black_Measles)'
    elif ClasesNo == 13: return 'Grape___healthy'
    elif ClasesNo == 14: return 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'
    elif ClasesNo == 15: return 'Orange___Haunglongbing_(Citrus_greening)'
    elif ClasesNo == 16: return 'Peach___Bacterial_spot'
    elif ClasesNo == 17: return 'Peach___healthy'
    elif ClasesNo == 18: return 'Pepper,_bell___Bacterial_spot'
    elif ClasesNo == 19: return 'Pepper,_bell___healthy'
    elif ClasesNo == 20: return 'Potato___Early_blight'
    elif ClasesNo == 21: return 'Potato___healthy'
    elif ClasesNo == 22: return 'Potato___Late_blight'
    elif ClasesNo == 23: return 'Raspberry___healthy'
    elif ClasesNo == 24: return 'Soybean___healthy'
    elif ClasesNo == 25: return 'Squash___Powdery_mildew'
    elif ClasesNo == 26: return 'Strawberry___healthy'
    elif ClasesNo == 27: return 'Strawberry___Leaf_scorch'
    elif ClasesNo == 28: return 'Tomato___Bacterial_spot'
    elif ClasesNo == 29: return 'Tomato___Early_blight'
    elif ClasesNo == 30: return 'Tomato___healthy'
    elif ClasesNo == 31: return 'Tomato___Late_blight'
    elif ClasesNo == 32: return 'Tomato___Leaf_Mold'
    elif ClasesNo == 33: return 'Tomato___Septoria_leaf_spot'
    elif ClasesNo == 34: return 'Tomato___Spider_mites Two-spotted_spider_mite'
    elif ClasesNo == 35: return 'Tomato___Target_Spot'
    elif ClasesNo == 36: return 'Tomato___Tomato_mosaic_virus'
    elif ClasesNo == 37: return 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    else: return 'Pertenece a una clase desconocida'

while True:
    #Leer imagenes
    ret, frame = captura.read()

    #Procesamiento de la imagen
    img = np.asarray(frame)
    img = cv2.resize(img, (224,224))
    img = preprocesamiento(img)
    #cv2.imshow("Imagen Preprocesada", img)
    img = img.reshape(1,224,224,3)
    cv2.putText(frame, "Clase: ", (20, 35), font, 0.75,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame, "Probabilidad:    ", (20,75), font, 0.75,(255,0,0), 2,cv2.LINE_AA)
    #Predecir imagen
    predicciones = model1.predict(img)
    classIndex =np.argmax(predicciones)
    valorDeProbabilidad = np.amax(predicciones)
    if valorDeProbabilidad > threshold:
        cv2.putText(frame, str(classIndex)+' '+str(getNombreClases(classIndex)),(120,35), font, 0.75,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(frame, str(round(valorDeProbabilidad*2,2))+'   '+' %',(100,75), font, 0.75,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('Resultado',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()       


