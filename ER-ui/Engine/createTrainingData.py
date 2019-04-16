import numpy as np
import os
import pandas as pd
import sys
import pickle




' Change directorys to folder with data'
'Parameters'
DATADIR = "A:\Computer science files\Disertation\Data"
FacesDIR = "A:\Computer science files\PycharmProjects\FinalProject"
sys.path.insert(0, DATADIR)
os.chdir(DATADIR)
MODELPATH = 'A:\Computer science files\Disertation\model2.h5'
width, height = 48, 48
neuron_numb = 64
num_labels = 7
batch_size = 64
epochs = 100

'Read in data'
data = pd.read_csv('fer2013.csv')
'Convert pixels column into a list '
pixels = data['pixels'].tolist()

faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)

    faces.append(face.astype('float32'))

sys.path.insert(0, FacesDIR)
os.chdir(FacesDIR)

fileObject = open("facesdatabase", 'wb')
pickle.dump(faces, fileObject)
fileObject.close()


emotionData = data['emotion'].tolist()
fileObject2 = open("emotionsfile", 'wb')
pickle.dump(emotionData, fileObject2)
fileObject2.close()



