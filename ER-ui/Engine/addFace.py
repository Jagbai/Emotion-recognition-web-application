import cv2
import sys
import numpy as np
import pickle
import os
import pandas as pd
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
FacesDIR = "C:/ER-ui/Engine"
sys.path.insert(0, FacesDIR)
os.chdir(FacesDIR)

'''
command line args: imgfile, name_of_person_in_image
'''

# todo display faces while enc
s,image,emotion = sys.argv

img = cv2.imread(image, 0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face = face_cascade.detectMultiScale(img, 1.3, 5)




for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    roi_img = img[y:y + h, x:x + w]
    roi_img = (np.expand_dims(cv2.resize(roi_img, (48, 48)), -1))
    facearr = np.asarray(roi_img).reshape(48,48)


'''
Code to open and append a face to the facesdatabase and then close it
'''
fileObjectop = open('facesdatabase', 'rb')
faces = pickle.load(fileObjectop)
faces.append(facearr.astype('float32'))
fileObjectcl = open("facesdatabase2", 'wb')
pickle.dump(faces, fileObjectcl)
fileObjectcl.close()

'''
Code to open and append an emotion to the emotions database and then close it
'''
fileObject2op = open("emotionsfile",'rb')
emotionData = pickle.load(fileObject2op)
emotionData.append(emotion)
fileObject2cl = open("emotionsfile2", 'wb')
pickle.dump(emotionData, fileObject2cl)
fileObject2cl.close()
