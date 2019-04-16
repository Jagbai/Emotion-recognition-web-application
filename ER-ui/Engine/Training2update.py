import numpy as np
import os
import pandas as pd
import sys
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator




' Change directorys to folder with data'
'Parameters'
FacesDIR = "A:/ER-ui/Engine"
MODELPATH = 'A:/ER-ui/Engine/Data/model2.h5'


width, height = 48, 48
neuron_numb = 64
num_labels = 7
batch_size = 64
epochs = 100

fileObject = open("facesfile",'rb')
faces = pickle.load(fileObject)
fileObject2 = open("emotionsfile",'rb')
emotionData = pickle.load(fileObject2)

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

emotionData = pd.Series(emotionData)
emotions = pd.get_dummies(emotionData).as_matrix()


X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
'Split training and testing data by 0.1 from faces and emotions'
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)
'split training data into both training and validation data'

NAME = "EMOTION-CLASSIFIER-MODEL-{}".format(int(time.time()))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.device('/cpu:0')
sess = tf.Session(config=config)


dense_layers = [2,3,4]
neuron_numbers = [32, 64, 128]
conv_layers = [2,3,4]

for dense_layer in dense_layers:
    for neuron_number in neuron_numbers:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-neurons-{}-dense-{}".format(conv_layer, neuron_number,dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(neuron_numb, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))


            for c in range(conv_layer):
                multiplierc = 2 ** c
                model.add(Conv2D(multiplierc * neuron_numb, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(BatchNormalization())
                model.add(Conv2D(multiplierc * neuron_numb, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                model.add(Dropout(0.5))

            model.add(Flatten())

            for d in range(dense_layer-1, -1,-1):
                multiplierd = 2 ** d
                model.add(Dense(multiplierd * neuron_numb, activation='relu', kernel_regularizer=l2(0.01)))
                model.add(Dropout(0.4))

            model.add(Dense(num_labels, activation='softmax'))

            model.summary()

            model.compile(loss=categorical_crossentropy,
                          optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                          metrics=['accuracy'])

            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
            MODELPATH = 'A:\Computer science files\Disertation\{}'.format(NAME)
            checkpointer = ModelCheckpoint(MODELPATH, monitor='val_loss', verbose=1, save_best_only=True)

            datagen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

            model.fit_generator(datagen.flow(np.array(X_train),np.array(y_train),
                      batch_size=batch_size), steps_per_epoch=len(X_train) / 32,
                      epochs=epochs,verbose=1,
                      validation_data=(np.array(X_test), np.array(y_test)),
                      shuffle=True,
                      callbacks=[lr_reducer,tensorboard, early_stopper,checkpointer])

            scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
            print("Loss: " + str(scores[0]))
            print("Accuracy: " + str(scores[1]))
