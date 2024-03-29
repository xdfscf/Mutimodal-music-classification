import os
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import app.models as models
from PIL import Image
from app import db
spectro_save_dir='./in/mel-specs/'
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.layers import *
from keras import initializers
import keras
from keras import backend as K
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from keras.models import Model
import ast
import tensorflow as tf
from CRNN_models import website_CRNN
def generate_spectrogram(signal, sr, n_fft, hop_length, music, spectrogram_id):
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    spectr_fname = music.music_name + str(spectrogram_id) + '.png'
    spectr_subdir = spectro_save_dir + spectr_fname[:2] + '/'

    if not os.path.exists(spectr_subdir):
        os.makedirs(spectr_subdir)
    subdir_path = spectr_subdir
    min_ = np.min(log_S)
    max_ = np.max(log_S)
    GI = (255 * (log_S - min_) / (max_ - min_)).astype(np.uint8)
    # Draw log values matrix in grayscale
    width = GI.shape[1]
    height = GI.shape[0]

    GI = np.asarray(GI).reshape(height, width, -1)[:, :646, :] \
        .reshape(height, -1)
    print(GI.shape)
    '''
    if len(models.Music_and_spectrogram_relation.query.filter_by(spectrogram_file=subdir_path + spectr_fname).all())==0:
        '''
    try:
        Image.fromarray(GI, 'L').save(subdir_path + spectr_fname)
        '''
        new_spectrogram=models.Music_and_spectrogram_relation(spectrogram_file=subdir_path + spectr_fname, music_id=music.id)
        db.session.add(new_spectrogram)
        db.session.commit()
        '''
        return subdir_path + spectr_fname
    except Exception:
        return None
        pass



def music_extract():
    music_list=[]
    for music in models.Music.query.filter((models.Music.valid==True)&(models.Music.label!=None)).all():
        music_list.append(music)
    return music_list


def music_clip():
    music_list=music_extract()
    for music in music_list:
        full_signal, sr = librosa.load(music.audio_file_name)
        duration = 30
        sample_len=full_signal.shape[0]
        n_samples = int(sr * (duration+5))
        if sample_len<=n_samples:
            music.valid = False
            db.session.commit()
        else:
            generate_spectrogram(signal = full_signal[sr * 5: sr * (duration+5)], sr=sr, n_fft=2048, hop_length=1024, music=music , spectrogram_id=1)

        if sample_len>=2*n_samples:
            generate_spectrogram(signal=full_signal[-sr * duration-5:-5], sr=sr, n_fft=2048, hop_length=1024, music=music,
                                 spectrogram_id=2)
        if sample_len>3*n_samples:
            generate_spectrogram(signal=full_signal[ sr * (duration+4) :sr * (duration*2+4)], sr=sr, n_fft=2048, hop_length=1024,
                                 music=music,
                                 spectrogram_id=3)
def music_train_data():
    spectrograms = models.train_data.query.all()
    images=[]
    labels=[]
    for spectrogram in spectrograms:
        images.append(np.asarray(Image.open(spectrogram.spectrogram_file).getdata()).reshape((128, 646)))
        labels.append(spectrogram.spectrogram_music.label)
        encoder = OneHotEncoder(handle_unknown='ignore')
    dict={}
    for i in labels:
        dict[i]=dict.get(i,0)+1
    print(dict)
    train_y = encoder.fit_transform(pd.DataFrame(labels)).toarray()
    train_x=np.array(images)
    return train_x, train_y

def CRNN_With_attention(img_height, img_width, output_size):
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

    x=LSTM(32, input_shape=(78, 64), return_sequences=True)(x)
    attention = Flatten()(x)
    attention = Dense(78, activation='tanh')(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(32)(attention)
    attention = Permute([2, 1])(attention)
    merge_model = keras.layers.Multiply()([x, attention])
    merge_model = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(32,), name='merge')(merge_model)
    output = Dense(output_size, activation='softmax', name='output')(merge_model)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
'''
music_clip()
'''
class My_Custom_Generator(keras.utils.all_utils.Sequence):

    def __init__(self, batch_size, block_number, type):
        self.batch_size = batch_size
        self.block_number = block_number
        self.type = type
        if self.type == "train":
            self.length = len(models.Text_train_data.query.all()) - len(
                models.Text_train_data.query.filter_by(block=self.block_number).all())
        elif self.type == "validate":
            self.length = len(models.Text_train_data.query.filter_by(block=self.block_number).all())
        db.session.remove()
    def __len__(self):
        return (np.ceil(self.length / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        '''
        train_x=[[],[],[],[],[]]
        data= models.Text_train_data.query.filter((models.Text_train_data.id>=idx * self.batch_size)&(models.Text_train_data.id<(idx+1) * self.batch_size)).all()
        batch_x = [i.text_data.split(' [seq] ')[:5] for i in data]
        batch_y = [ast.literal_eval(i.one_hot_encoding) for i in data]
        for paragraph in batch_x:
            for index in range(len(paragraph)):
                train_x[index].append(paragraph[index])
        train_x = [tf.constant(i, dtype=tf.string) for i in train_x]
        return train_x, np.asarray(batch_y).astype('float64')
        '''
        if self.type=="train":
            data = models.Text_train_data.query.filter((models.Text_train_data.block!=self.block_number)|(models.Text_train_data.block==None)).all()
        elif self.type == "validate":
            data = models.Text_train_data.query.filter_by(block=self.block_number).all()
        data=data[idx * self.batch_size:min(len(data), (idx + 1) * self.batch_size)]

        batch_y=[]
        images=[]
        for i in data:
            fpath = i.spectrogram_file
            try:
                images.append(np.asarray(Image.open(fpath).getdata()).reshape((128, 646)))
                batch_y.append(ast.literal_eval(i.one_hot_encoding))
            except Exception:
                pass

        images = np.array(images).astype('float32')
        images = images.reshape((images.shape[0], 128, 646, 1))

        db.session.remove()
        return images, np.asarray(batch_y).astype('float32')

if __name__ == "__main__":

    model = keras.models.load_model("validate0-improvement-234-0.98.hdf5")
    '''
    model2 = Model(model.input, model.get_layer('lambda').output)
    model2.save('./CRNN_ATTENTION_FEATURE_32.h5')
    '''
    train_generator = My_Custom_Generator(20, 0, "train")
    valid_generator = My_Custom_Generator(20, 0, "validate")
    filepath = "validate0-2-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.001)
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, learning_rate_reduction]

    history = model.fit_generator(generator=train_generator,validation_data = valid_generator, steps_per_epoch=60, epochs=250, verbose=1, callbacks=callbacks_list)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('CRNN-A validate 0 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    model.save('./final_validate0-2.hdf5')
