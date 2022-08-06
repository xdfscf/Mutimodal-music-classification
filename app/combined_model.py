from PIL import Image
from keras.layers import *
import ast
import keras
from matplotlib import pyplot as plt

import app.models as models
from app import db
import numpy as np
from sqlalchemy import func
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.preprocessing import OneHotEncoder
from official.nlp import optimization
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
from keras import backend as K

def encoder():

    audio_model= keras.models.load_model("CRNN_ATTENTION_FEATURE_32.h5", custom_objects={"K": K}, )
    for layer in audio_model.layers:
        layer._name=layer._name+"_2"

    lyric_model= keras.models.load_model("Bert_FEATURE_512.h5",custom_objects={'KerasLayer':hub.KerasLayer})
    print(lyric_model.output.shape)
    print(audio_model.output.shape)
    audio_model.trainable=False
    merged_output= Concatenate()([audio_model.output, lyric_model.output])

    model = Model(inputs=[audio_model.input, lyric_model.input], outputs=merged_output, name="classifier")

    return model

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors

        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )


        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def custom_loss(labels, feature_vectors):
    # calculate loss, using y_pred
    feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
    # Compute logits
    logits = tf.divide(
        tf.matmul(
            feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
        ),
        0.05,
    )

    return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def add_projection_head(encoder):
    inputs1 =Input(shape=(128,646, 1), name='audio_inputs')
    inputs2=Input(shape=(),dtype=tf.string, name='lyric_input')
    features = encoder([inputs1, inputs2])
    outputs = Dense(128, activation="relu", name="denses")(features)
    outputs= Dense(10, activation="softmax" ,name="combine_output")(outputs)
    model = keras.Model(
        inputs=[inputs1, inputs2], outputs=outputs, name="encoder_with_projection-head"
    )
    return model

def outer_product():
    audio_model = keras.models.load_model("CRNN_ATTENTION_FEATURE_32.h5", custom_objects={"K": K}, )
    for layer in audio_model.layers:
        layer._name = layer._name + "_2"

    lyric_model = keras.models.load_model("Bert_FEATURE_512.h5", custom_objects={'KerasLayer': hub.KerasLayer})

    bias1 = tf.ones((tf.shape(audio_model.output)[0], 1))
    bias1.trainable=False
    bias2 = tf.ones((tf.shape(lyric_model.output)[0], 1))
    bias2.trainable=False
    merged_output1 = Concatenate()([bias1, audio_model.output])
    merged_output2 = Concatenate()([bias2, lyric_model.output])
    print(merged_output1.shape)
    print(merged_output2.shape)
    merged=tf.einsum('ki,kj->kij',merged_output1,merged_output2)
    merged=Flatten(name="flatten_merge")(merged)
    output=Dense(10, activation="softmax", name="final_output")(merged)

    model = Model(inputs=[audio_model.input, lyric_model.input], outputs=output, name="classifier")

    return model

'''
class My_Custom_Generator(keras.utils.all_utils.Sequence):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.lenght=len(models.Text_train_data.query.all())
        db.session.remove()
    def __len__(self):
        return (np.ceil(self.lenght / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        
        train_x=[[],[],[],[],[]]
        data= models.Text_train_data.query.filter((models.Text_train_data.id>=idx * self.batch_size)&(models.Text_train_data.id<(idx+1) * self.batch_size)).all()
        batch_x = [i.text_data.split(' [seq] ')[:5] for i in data]
        batch_y = [ast.literal_eval(i.one_hot_encoding) for i in data]
        for paragraph in batch_x:
            for index in range(len(paragraph)):
                train_x[index].append(paragraph[index])
        train_x = [tf.constant(i, dtype=tf.string) for i in train_x]
        return train_x, np.asarray(batch_y).astype('float64')
        

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
'''
class My_Custom_Generator(keras.utils.all_utils.Sequence):

    def __init__(self, batch_size, block_number, type, metrix):
        self.batch_size = batch_size
        self.block_number=block_number
        self.type=type
        self.metrix=metrix
        if self.type=="train":
            self.length =len(models.Text_train_data.query.all())-len(models.Text_train_data.query.filter_by(block=self.block_number).all())
        elif self.type=="validate":
            self.length =len(models.Text_train_data.query.filter_by(block=self.block_number).all())
        db.session.remove()
    def __len__(self):
        return ((np.ceil(self.length / float(self.batch_size)))).astype(np.int)

    def __getitem__(self, idx):

        if self.type=="train":
            data = models.Text_train_data.query.filter((models.Text_train_data.block!=self.block_number)|(models.Text_train_data.block==None)).all()
        elif self.type == "validate":
            data = models.Text_train_data.query.filter_by(block=self.block_number).all()
        '''
        if idx* self.batch_size>len(data):
            idx=random.randint(0,np.ceil(self.length / float(self.batch_size)))
        '''
        if min(len(data), (idx + 1) * self.batch_size)-idx * self.batch_size<20:
            data = data[-20:]
        else:
            data = data[idx * self.batch_size:min(len(data), (idx + 1) * self.batch_size)]
        batch_x = [' [sep] '.join(i.text_data.split(' [seq] ')) for i in data]
        if self.metrix == "contrastive":
            batch_y = [i.label for i in data]
        elif self.metrix == "cross_entropy":
            batch_y = [ast.literal_eval(i.one_hot_encoding) for i in data]
        images = []
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

def get_data(block_number,type):
    if type == "train":
        data = models.Text_train_data.query.filter(
            (models.Text_train_data.block != block_number) | (models.Text_train_data.block == None)).all()
    elif type == "validate":
        data = models.Text_train_data.query.filter_by(block=block_number).all()
    batch_x = [' [sep] '.join(i.text_data.split(' [seq] ')) for i in data]
    batch_y = [ast.literal_eval(i.one_hot_encoding) for i in data]
    images = []
    for musics in data:
        fpath = musics.spectrogram_file
        images.append(np.asarray(Image.open(fpath).getdata()).reshape((128, 646)))

    images = np.array(images).astype('float64')
    images = images.reshape((images.shape[0], 128, 646, 1))
    db.session.remove()

    return [images, tf.constant(batch_x, dtype=tf.string)], np.asarray(batch_y).astype('float64')
'''
encoder = encoder()

encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
   optimizer="Adam",
   loss=custom_loss,
)
encoder_with_projection_head.summary()
train_generator=My_Custom_Generator(20,4,"train","contrastive")
valid_generator=My_Custom_Generator(20,4,"validate", "contrastive")
filepath="contrastive-combine-{epoch:02d}-{loss:.2f}.h5"
encoder_with_projection_head.fit_generator(generator=train_generator,validation_data = valid_generator,steps_per_epoch =60,epochs=150,verbose=1)
encoder_with_projection_head.save("./my_model.hdf5")


encoder_with_projection_head.trainable=False
dense = Flatten()(encoder_with_projection_head.get_layer('denses').output)
dense=Dense(64, activation='relu', name="projection")(dense)
output = Dense(10, activation='softmax', name="merged_output")(dense)

new_model = keras.Model(inputs=[encoder_with_projection_head.get_layer('audio_inputs').input,encoder_with_projection_head.get_layer('lyric_input').input], outputs=output)
new_model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

'''

encoder = encoder()

new_model = add_projection_head(encoder)
new_model.compile(
   optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
)



filepath="validate3-combine-cont-{epoch:02d}-{val_accuracy:.2f}.hdf5"
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.001)
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, learning_rate_reduction]
train_generator=My_Custom_Generator(20,3,"train","cross_entropy")
valid_generator=My_Custom_Generator(20,3,"validate","cross_entropy")
history=new_model.fit_generator(generator=train_generator,validation_data = valid_generator,steps_per_epoch =60,epochs=250,verbose=1,callbacks=callbacks_list)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Combine model validate 3 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
'''
train_x,train_y=get_data(4,"train")
new_model.evaluate(train_x,train_y)
train_x,train_y=get_data(4,"validate")
new_model.evaluate(train_x,train_y)
'''
new_model.save('./final_combine3.hdf5')
