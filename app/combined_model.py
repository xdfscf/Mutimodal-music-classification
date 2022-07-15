from PIL import Image
from keras.layers import *
import ast
import keras
import app.models as models
from app import db
import numpy as np
from sqlalchemy import func
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from official.nlp import optimization
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.callbacks import ModelCheckpoint
def encoder(output_size):
    audio_model= keras.models.load_model("CRNN_ATTENTION_FEATURE_32.h5")

    lyric_model= keras.models.load_model("Bert_FEATURE_512.h5",custom_objects={'KerasLayer':hub.KerasLayer})

    merged= Concatenate()([audio_model.output, lyric_model.output['pooled_output']])
    output= Dense( output_size,name='dense_1', activation="softmax")(merged)
    model = Model(inputs=[audio_model.input, lyric_model.input], outputs=output, name="cifar10-classifier")
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

class My_Custom_Generator(keras.utils.all_utils.Sequence):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.lenght=len(models.Text_train_data.query.all())
        db.session.remove()
    def __len__(self):
        return (np.ceil(self.lenght / float(self.batch_size))).astype(np.int)

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

        data = models.Text_train_data.query.filter((models.Text_train_data.id >= idx * self.batch_size) & (
                    models.Text_train_data.id < (idx + 1) * self.batch_size)).all()

        batch_x = [' [sep] '.join(i.text_data.split(' [seq] ')) for i in data]
        batch_y = [ast.literal_eval(i.one_hot_encoding) for i in data]
        images=[]
        for musics in data:
            fpath = musics.spectrogram_file
            images.append(np.asarray(Image.open(fpath).getdata()).reshape((128, 646)))
        images = np.array(images).astype('float64')
        images = images.reshape((images.shape[0], 128, 646, 1))
        db.session.remove()
        return [images, tf.constant(batch_x, dtype=tf.string)], np.asarray(batch_y).astype('float64')

def data_clean():
    data = models.Text_train_data.query.filter().all()
    for i in data:
        try:
            fpath = i.spectrogram_file
            np.asarray(Image.open(fpath).getdata()).reshape((128, 646))
        except Exception:
            print("delete ", i)
            db.session.delete(i)
            db.session.commit()
data_clean()
model=encoder(7)
learning_rate_reduction=ReduceLROnPlateau(monitor='accuracy',patience=3,verbose=1,factor=0.5, min_lr=0.001)
train_generator=My_Custom_Generator(10)
filepath="weights15-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint, learning_rate_reduction]

model.fit_generator(generator=train_generator,steps_per_epoch =60,epochs=250,verbose=1, callbacks=callbacks_list)
