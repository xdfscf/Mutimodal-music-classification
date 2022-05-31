from keras import initializers
import scipy.misc
import keras
from keras import Sequential

from keras.layers import *
import numpy as np

import time

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

import data

def VCG_net(img_height, img_width, output_size):
    # CNN(VGG)
    initializer = initializers.he_normal()
    model = Sequential()
    model.add(Input(shape=(img_height, img_width, 1), name='img_inputs'))

    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name='conv1'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1'))
    model.add(Dropout(0.1, name='dropout1'))

    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool2'))
    model.add(Dropout(0.1, name='dropout2'))

    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv3'))
    model.add(BatchNormalization(name='bn3'))
    model.add(ELU())
    model.add(Dropout(0.1, name='dropout3'))

    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv4'))
    model.add(BatchNormalization(name='bn4'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool3'))
    model.add(Dropout(0.1, name='dropout4'))

    model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name='conv5'))
    model.add(BatchNormalization(name='bn5'))
    model.add(ELU())

    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv6'))
    model.add(BatchNormalization(name='bn6'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool4'))

    model.add(Conv2D(128, (2, 2), padding="valid", activation='relu', kernel_initializer=initializer, name='conv7'))
    model.add(BatchNormalization(name='bn7'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), name="conv_output"))

    model.add(Reshape(target_shape=(240, 128)))

    model.add(GRU(64, return_sequences=True, name='gru1'))
    model.add(GRU(64, return_sequences=False, name='gru2'))

    model.add(Dropout(0.3, name='dropout_final'))

    model.add(Dense(output_size, activation='softmax', name='output'))

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

data = data.get_data()
x_train, y_train=data.train.load_imgae_with_onehot_labels()
model = VCG_net(128,646, len(y_train[0]))
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5, min_lr=0.00001)
print(x_train.shape,y_train.shape)
x_train=x_train.reshape((3893,128,646,1))
model.fit(np.asarray(x_train).astype('float32'),np.asarray(y_train).astype('float32'), batch_size=20,epochs=10,verbose=1, callbacks=[learning_rate_reduction])

