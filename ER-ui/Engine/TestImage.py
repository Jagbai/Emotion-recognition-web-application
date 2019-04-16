import numpy as np
import cv2
import sys
import tensorflow as tf
from keras.models import load_model


import h5py

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

MODELPATH = 'C:/ER-ui/Engine/Data/model.h5'
model = load_model(MODELPATH)
predictedemote = "Face not found"
s,image = sys.argv
img = image
img = cv2.imread(img, 0)

face_cascade = cv2.CascadeClassifier('C:/ER-UI/Engine/haarcascade_frontalface_default.xml')
face = face_cascade.detectMultiScale(img, 1.3, 5)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    roi_gray = img[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    prediction = model.predict(cropped_img)
    predictedemote = emotion_dict[int(np.argmax(prediction))]

print(predictedemote)
sys.stdout.flush()

