from keras import initializers
import keras
from keras import Sequential
from keras import backend as K
from keras.layers import *
import numpy as np
from keras.callbacks import ModelCheckpoint
from PIL import Image
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers
import tensorflow as tf

def CRNN_With_attention(img_height, img_width, output_size):
    # CNN(VGG)
    initializer = initializers.he_normal()

    input=Input(shape=(img_height, img_width, 1), name='img_inputs')
    x = ZeroPadding2D(padding=(0, 37))(input)
    x = BatchNormalization(axis=2, name='bn_0_freq')(x)
    x=Conv2D(32, (3, 3), padding="same", kernel_initializer=initializer, name='conv1')(x)
    x=BatchNormalization()(x)
    x=ELU()(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')(x)
    x=Dropout(0.1, name='dropout1')(x)

    x=Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name='conv2')(x)
    x=BatchNormalization(name='bn2')(x)
    x=ELU()(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool2')(x)
    x=Dropout(0.1, name='dropout2')(x)

    x=Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv3')(x)
    x=BatchNormalization(name='bn3')(x)
    x=ELU()(x)
    x=Dropout(0.1, name='dropout3')(x)

    x=Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv4')(x)
    x=BatchNormalization(name='bn4')(x)
    x=ELU()(x)
    x=MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool3')(x)
    x=Dropout(0.1, name='dropout4')(x)

    x=Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv5')(x)
    x=BatchNormalization(name='bn5')(x)
    x=ELU()(x)

    x=Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv6')(x)
    x=BatchNormalization(name='bn6')(x)
    x=ELU()(x)
    x=MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool4')(x)

    x=Conv2D(128, (2, 2), padding="valid", activation='relu', kernel_initializer=initializer, name='conv7')(x)
    x=BatchNormalization(name='bn7')(x)
    x=ELU()(x)

    x=MaxPooling2D(pool_size=(2, 2), name="conv_output")(x)

    x=Conv2D(64, (3, 3), padding="valid", activation='relu', kernel_initializer=initializer, name='conv8')(x)
    x=BatchNormalization(name='bn8')(x)
    x=ELU()(x)

    x=Reshape(target_shape=(87, 64))(x)

    attention = Flatten()(x)
    attention = Dense(87, activation='tanh')(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(64)(attention)
    attention = Permute([2, 1])(attention)
    merge_model = keras.layers.Multiply()([x, attention])
    merge_model = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(64,))(merge_model)
    output = Dense(output_size, activation='softmax', name='output')(merge_model)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def website_CRNN(img_height, img_width, output_size):
    '''

    128, 646+37=683
    '''
    channel_axis = 1
    freq_axis = 2
    time_axis = 3
    input = Input(shape=(img_height, img_width, 1), name='img_inputs')
    x = ZeroPadding2D(padding=(0, 37))(input)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)

    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)

    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)

    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)

    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 2), strides=(2, 2), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)
    '''
    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)

    '''
    attention = Flatten()(x)
    attention = Dense(15, activation='tanh')(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)
    merge_model = keras.layers.Multiply()([x, attention])
    merge_model=layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(128,))(merge_model)



    output = Dense(output_size, activation='softmax', name='output')(merge_model)

    # Create model
    model = keras.Model(input, output)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
def CRNN_With_attention2(img_height, img_width, output_size):
    # CNN(VGG)
    initializer = initializers.he_normal()

    input=Input(shape=(img_height, img_width, 1), name='img_inputs')

    x=Conv2D(32, (3, 3), padding="same", kernel_initializer=initializer, name='conv1')(input)
    x=BatchNormalization()(x)
    x=ELU()(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')(x)
    x=Dropout(0.1, name='dropout1')(x)

    x=Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name='conv2')(x)
    x=BatchNormalization(name='bn2')(x)
    x=ELU()(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool2')(x)
    x=Dropout(0.1, name='dropout2')(x)

    x=Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv3')(x)
    x=BatchNormalization(name='bn3')(x)
    x=ELU()(x)
    x=Dropout(0.1, name='dropout3')(x)

    x=Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv4')(x)
    x=BatchNormalization(name='bn4')(x)
    x=ELU()(x)
    x=MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool3')(x)
    x=Dropout(0.1, name='dropout4')(x)

    x=Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv5')(x)
    x=BatchNormalization(name='bn5')(x)
    x=ELU()(x)

    x=Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv6')(x)
    x=BatchNormalization(name='bn6')(x)
    x=ELU()(x)
    x=MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool4')(x)

    x=Conv2D(128, (2, 2), padding="valid", activation='relu', kernel_initializer=initializer, name='conv7')(x)
    x=BatchNormalization(name='bn7')(x)
    x=ELU()(x)

    x=MaxPooling2D(pool_size=(2, 2), name="conv_output")(x)

    x=Conv2D(64, (3, 3), padding="valid", activation='relu', kernel_initializer=initializer, name='conv8')(x)
    x=BatchNormalization(name='bn8')(x)
    x=ELU()(x)

    x=Reshape(target_shape=(78, 64))(x)
    x=LSTM(64, input_shape=(78, 64))(x)


    attention = Flatten()(x)
    attention = Dense(78, activation='tanh')(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(64)(attention)
    attention = Permute([2, 1])(attention)
    merge_model = keras.layers.Multiply()([x, attention])
    merge_model = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(64,))(merge_model)
    output = Dense(output_size, activation='softmax', name='output')(merge_model)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
def CRNN(img_height, img_width, output_size):
    # CNN(VGG)
    initializer = initializers.he_normal()
    model = Sequential()
    model.add(Input(shape=(img_height, img_width, 1), name='img_inputs'))

    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=initializer, name='conv1'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1'))
    model.add(Dropout(0.1, name='dropout1'))

    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name='conv2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool2'))
    model.add(Dropout(0.1, name='dropout2'))

    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv3'))
    model.add(BatchNormalization(name='bn3'))
    model.add(ELU())
    model.add(Dropout(0.1, name='dropout3'))

    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv4'))
    model.add(BatchNormalization(name='bn4'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool3'))
    model.add(Dropout(0.1, name='dropout4'))

    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='conv5'))
    model.add(BatchNormalization(name='bn5'))
    model.add(ELU())

    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='conv6'))
    model.add(BatchNormalization(name='bn6'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='maxpool4'))

    model.add(Conv2D(128, (2, 2), padding="valid", activation='relu', kernel_initializer=initializer, name='conv7'))
    model.add(BatchNormalization(name='bn7'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), name="conv_output"))

    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu', kernel_initializer=initializer, name='conv8'))
    model.add(BatchNormalization(name='bn8'))
    model.add(ELU())

    model.add(Reshape(target_shape=(78, 128)))

    model.add(LSTM(32,input_shape=(78, 128)))

    model.add(Dropout(0.3, name='dropout_final'))

    model.add(Dense(output_size, activation='softmax', name='output'))

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model