from __future__ import print_function

import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd


# Refer for index values (lua arrays are 1-indexed, so this is 1 less)
# classId, track, file = r[8], r[0], r[1]
# For labels
#dataset[idx][8]

# Load the saved numpy image here
X_train = np.load('data/X_train_48.npy')
Y_train = np.load('data/Y_train.npy')
X_valid = np.load('data/X_valid_48.npy')
Y_valid = np.load('data/Y_valid.npy')


# Running a simple convnet model on the original dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model


batch_size = 128
nb_classes = 43
data_augmentation = True
nb_epoch = 50
data_augmentation = False
img_rows, img_cols = 32, 32
# the images are RGB
img_channels = 3
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

# Build Idsia Sequential Model
"""
model = Sequential()

model.add(Convolution2D(100, 7, 7, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(150, 4, 4, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(250, 4, 4))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
"""

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])


if not data_augmentation:
	model.fit(X_train, Y_train,
    	          batch_size=batch_size,
        	      nb_epoch=nb_epoch,
            	  validation_data=(X_valid, Y_valid),
	          	  verbose=1,
               	  shuffle=True)

else:
	datagen = ImageDataGenerator(
        	featurewise_center=False,  # set input mean to 0 over the dataset
        	samplewise_center=True,  # set each sample mean to 0
        	featurewise_std_normalization=False,  # divide inputs by std of the dataset
        	samplewise_std_normalization=True,  # divide each input by its std
        	zca_whitening=True,  # apply ZCA whitening
        	rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        	width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        	height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        	horizontal_flip=True,  # randomly flip images
        	vertical_flip=True)  # randomly flip images

	datagen.fit(X_train)

	# fit the model on the batches generated by datagen.flow()
    	model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_data=(X_test, Y_test))

model.save('models/simple_aug__model.h5')