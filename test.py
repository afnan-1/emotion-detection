'''
Afnan
Emotion Detection Using AI Deep Learning!
'''

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2 as opencv
import numpy as np

face_classifier = opencv.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = opencv.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    # convert image into gray scale image
    gray = opencv.cvtColor(frame,opencv.COLOR_BGR2GRAY)
    # detect our faces 
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    # 4 parameters in our face while detecting
    for (x,y,w,h) in faces:
        # rectangle shape in our face
        opencv.rectangle(frame,(x,y),(x+w,y+h),(255,1,1),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = opencv.resize(roi_gray,(48,48),interpolation=opencv.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class
# calculate maximum probability
            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            label_position = (x,y)
            opencv.putText(frame,label,label_position,opencv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            opencv.putText(frame,'No Face Found',(20,60),opencv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    opencv.imshow('Emotion Detector Project Sonia zulfiqar',frame)
    if opencv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
opencv.destroyAllWindows()


























