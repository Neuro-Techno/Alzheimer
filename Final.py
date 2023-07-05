from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import np_utils
import sklearn.model_selection as ms
from tensorflow.keras.layers import BatchNormalization
from keras.utils import to_categorical
import os
import cv2
import numpy as np
import pandas as pd

# reading images path

Mild_Demented_PATH = "/Dataset/Mild_Demented/"
Moderate_Demented_PATH = "/Dataset/Moderate_Demented/"
Non_Demented_PATH = "/Dataset/Non_Demented/"
Very_Mild_Demented_PATH = "/Dataset/Very_Mild_Demented/"

# sort inputs images to list

Mild_Demented_List = sorted(next(os.walk(Mild_Demented_PATH))[2])
Moderate_Demented_List = sorted(next(os.walk(Moderate_Demented_PATH))[2])

Non_Demented_List = sorted(next(os.walk(Non_Demented_PATH))[2])
Very_Mild_Demented_List = sorted(next(os.walk(Very_Mild_Demented_PATH))[2])

# making inputs list


def read_images_from_dir(dir_path):
    images = []
    for filename in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, filename))
        if img is not None:
            images.append(img)
    return images


images_nums = len(Mild_Demented_List)+len(Moderate_Demented_List) + \
    len(Non_Demented_List)+len(Very_Mild_Demented_List)

inputs = np.zeros((images_nums, 128, 128, 3), dtype=np.uint8)

images1 = read_images_from_dir(Mild_Demented_PATH)
inputs[:len(images1)] = images1

images2 = read_images_from_dir(Moderate_Demented_PATH)
inputs[len(images1):len(images1)+len(images2)] = images2

images3 = read_images_from_dir(Non_Demented_PATH)
inputs[len(images1)+len(images2):len(images1) +
       len(images2)+len(images3)] = images3

images4 = read_images_from_dir(Very_Mild_Demented_PATH)
inputs[len(images1)+len(images2)+len(images3):len(images1) +
       len(images2)+len(images3)+len(images4)] = images4

# normalizing

inputs = inputs.astype('float32')
inputs = inputs/255

# Mild_Demented_List  0
# Moderate_Demented_List  1
# Non_Demented_List 2
# Very_Mild_Demented_List 3

# making answers list then categorical that

masks = np.zeros((images_nums, 1))
masks[:len(Mild_Demented_List)] = 0
masks[len(Mild_Demented_List):len(Mild_Demented_List) +
      len(Moderate_Demented_List)] = 1
masks[len(Mild_Demented_List)+len(Moderate_Demented_List):len(Mild_Demented_List)+len(Moderate_Demented_List)+len(Non_Demented_List)] = 2
masks[len(Mild_Demented_List)+len(Moderate_Demented_List)+len(Non_Demented_List):len(Mild_Demented_List) +
      len(Moderate_Demented_List)+len(Non_Demented_List)+len(Very_Mild_Demented_List)] = 3

masks = np_utils.to_categorical(masks)

# spliting data

xtrain, xtest, ytrain, ytest = ms.train_test_split(
    inputs, masks, train_size=0.85, shuffle=True)

# Network design with keras

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
          input_shape=xtrain.shape[1:], activation='elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(MaxPool2D(2, 2))
model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(MaxPool2D(2, 2))
model.add(Conv2D(256, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(MaxPool2D(2, 2))
model.add(Conv2D(512, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())


model.add(Conv2D(1024, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())


model.add(Conv2D(512, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())


model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding='same',
          input_shape=xtrain.shape[1:], activation='elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(32, activation='elu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(xtrain, ytrain, validation_data=(
    xtest, ytest), epochs=30, batch_size=4)

# plot

pd.DataFrame(history.history).plot()
