from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2
from tensorflow.keras import datasets, layers, models

import tensorflow as tf

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

Size=np.shape(img)
model1 = tf.keras.models.load_model('ModeloPlantVillage.h5')
objects = ('Costra de Manzana', 'apple black rot', 'Cedar_apple_rust','Apple___healthy','Blueberry___healthy'
            ,'Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew'
            ,'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust','Corn_(maize)___healthy'
            ,'Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___healthy'
            ,'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot'
            ,'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight'
            ,'Potato___healthy','Potato___Late_blight','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew'
            ,'Strawberry___healthy','Strawberry___Leaf_scorch','Tomato___Bacterial_spot','Tomato___Early_blight'
            ,'Tomato___healthy', 'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot'
            ,'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_mosaic_virus'
            ,'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Clase desconocida')
            
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (228,200,50)
lineType               = 2

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
#    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame = cv2.rectangle(frame, (100,100), (200,200), (255, 0, 0),2)
    I1 = frame[100:200,100:200,:]
    I = cv2.resize(I1, (224,224), interpolation = cv2.INTER_AREA)
    I = np.double(I)/255
    I = I[np.newaxis, ...]
    answer = model1.predict(I)
    answer = np.array(answer).ravel()
    x=np.argmax(answer)
    if np.max(answer)>0.9: # Nivel de confianza para la deteccion [entre -1 y 1]
        cv2.putText(frame,objects[x], 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
    
#    
#    img_out = LBP(img, 3)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('ROI',I1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()