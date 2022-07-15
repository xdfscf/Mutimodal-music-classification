import ast
import copy
import datetime
import random
from xmlrpc.client import DateTime

import keras
import librosa
import requests

from music_data_preprocessing import generate_spectrogram
import app.models as models
from app import db
import time
import _thread
import re
import csv

import os
import csv
import operator
import re
import pandas as pd

import requests

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

def valid_data_check():
    file_list=[]
    music_dir = './audios'
    for subdir, _, files in os.walk(music_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                file_list.append('audios\\'+file)
    for music in models.Music.query.filter((models.Music.lyric!=None)&(models.Music.audio_file_name!=None)).all():
         if music.audio_file_name in file_list:
             music.valid=True
         else:
             music.valid=False
         db.session.commit()

def top_tag_select():
    tag_number_dic={}
    tag_entity_dic={}
    for i in models.Music_tag.query.all():
        tag_number_dic[i.tag_name]=len(i.tag_s_musics.all())
        tag_entity_dic[i.tag_name] =i.tag_s_musics.filter(models.Music.valid==True).all()
        '''
        print(i.tag_s_musics.filter(models.Music.valid==True).all())
        '''
    tag_dic=dict(sorted(tag_number_dic.items(), key=lambda item: item[1], reverse = True))
    print(tag_dic)
    print(tag_entity_dic)
    keys=list(tag_dic.keys())
    tag_index_dic={}


    intersection_dic={}

    tag_dominate_dic={}
    tag_set=set([])
    sub_dic1={}
    sub_dic2={}
    dominate_tag=['rock', 'pop', 'electronic', 'soul', 'punk', 'blues','country', 'folk', 'heavy metal','jazz']
    dominate_id=[]
    for i in dominate_tag:
        dominate_id.append(models.Music_tag.query.filter_by(tag_name=i).first().id)

    dominate=len(dominate_tag)
    for i in range(dominate):
        tag_index_dic[str(i)]=dominate_tag[i]
        tag_dominate_dic[str([i])]=tag_entity_dic[dominate_tag[i]]
        tag_set=tag_set.union(set(tag_entity_dic[keys[i]]))
        sub_dic1[str([i])]=copy.copy(tag_dominate_dic[str([i])])
        sub_dic2[str([i])]=copy.copy(tag_dominate_dic[str([i])])

    print(len(tag_set))
    print(tag_dominate_dic)
    '''
    tag_dominate_dic={'[0]':[1,2,3,4],'[1]':[2,3,4,5],'[2]':[3,4,5,6,7],'[3]':[1,4,2,6,8,9]}
    sub_dic1={'[0]':[1,2,3,4],'[1]':[2,3,4,5],'[2]':[3,4,5,6,7],'[3]':[1,4,2,6,8,9]}
    sub_dic2={'[0]':[1,2,3,4],'[1]':[2,3,4,5],'[2]':[3,4,5,6,7],'[3]':[1,4,2,6,8,9]}
    '''
    intersection_dic['0']=sub_dic1
    print(id(intersection_dic['0'][str([0])][0]),id(tag_dominate_dic[str([0])][0]))
    for i in range(1,dominate):
        intersection_sub_dic = {}
        for key in list(intersection_dic[str(i-1)].keys()):
            intersection_before=ast.literal_eval(key)
            if intersection_before[-1]<10:
                for j in range(intersection_before[-1]+1,dominate):
                    intersection_after=copy.deepcopy(intersection_before)
                    intersection_after.append(j)
                    if i==1:
                        intersection_after_entity = list(
                            set(intersection_dic[str(i - 1)][str(intersection_before)]).intersection(set(intersection_dic[str(i - 1)][str([j])])))
                        intersection_sub_dic[str(intersection_after)] = intersection_after_entity
                        for k in list(intersection_dic[str(i-1)].keys()):
                            intersection_dic[str(i - 1)][k] = list(set(intersection_dic[str(i - 1)][k]) - set(intersection_after_entity))

                    else:

                        intersection_after_entity = list(
                            set(intersection_dic[str(i - 1)][str(intersection_before)]).intersection(
                                set(tag_dominate_dic[str([j])])))
                        intersection_sub_dic[str(intersection_after)] = intersection_after_entity
                        intersection_dic[str(i - 1)][str(intersection_before)] = list(
                            set(intersection_dic[str(i - 1)][str(intersection_before)]) - set(intersection_after_entity))

        intersection_dic[str(i)]=intersection_sub_dic

    for inter_key in list(intersection_dic.keys()):
        for key in list(intersection_dic[inter_key].keys()):
            if len(intersection_dic[inter_key][key])!=0:
                print(key, len(intersection_dic[inter_key][key]), intersection_dic[inter_key][key])

    relation_query=models.Music_and_tag_relation.query.filter(models.Music_and_tag_relation.tag_id.in_(dominate_id))
    for inter_key in list(intersection_dic.keys()):
        for key in list(intersection_dic[inter_key].keys()):
            for music in intersection_dic[inter_key][key]:
                tag=db.session.query(func.max(models.Music_and_tag_relation.weight),models.Music_and_tag_relation.tag_id).filter(models.Music_and_tag_relation.tag_id.in_(dominate_id),models.Music_and_tag_relation.music_id==music.id).first()
                num_of_equal_weight_tag=len(relation_query.filter((models.Music_and_tag_relation.music_id==music.id)&(models.Music_and_tag_relation.weight==tag[0])).all())
                if num_of_equal_weight_tag==1:
                    music.label=tag.tag_id
                    db.session.commit()
                else:
                    music.valid=False
                    db.session.commit()


# Modify the dataframe df by converting all tweets to lower case.
def lower_case(string):
    string = string.lower()
    return string


def remove_non_alphabetic_chars(string):
    string = re.sub(r'[^A-Za-z]', ' ', string)
    return string

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(string):
    string = re.sub(r'\s+', ' ', string.strip())
    return string

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(string):
    string = re.split(' ', string)
    return string



def bert_models():
    bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
    map_name_to_handle = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
        'bert_en_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
        'bert_multi_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-2_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-2_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-2_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-4_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-4_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-4_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-4_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-6_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-6_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-6_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-6_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-8_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-8_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-8_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-8_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-10_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-10_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-10_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-10_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-12_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-12_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-12_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
        'albert_en_base':
            'https://tfhub.dev/tensorflow/albert_en_base/2',
        'electra_small':
            'https://tfhub.dev/google/electra_small/2',
        'electra_base':
            'https://tfhub.dev/google/electra_base/2',
        'experts_pubmed':
            'https://tfhub.dev/google/experts/bert/pubmed/2',
        'experts_wiki_books':
            'https://tfhub.dev/google/experts/bert/wiki_books/2',
        'talking-heads_base':
            'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
    }

    map_model_to_preprocess = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'bert_en_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'bert_multi_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
        'albert_en_base':
            'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
        'electra_small':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'electra_base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'experts_pubmed':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'experts_wiki_books':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'talking-heads_base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    }

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

    print(f'BERT model selected           : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
    bert_model = hub.KerasLayer(tfhub_handle_encoder)
    return bert_model, bert_preprocess_model, tfhub_handle_preprocess, tfhub_handle_encoder

def bert_attention_for_text(tfhub_handle_preprocess, tfhub_handle_encoder, output_size):
    text_input1 = tf.keras.layers.Input(shape=(),dtype=tf.string, name='sentences1')
    text_input2 = tf.keras.layers.Input(shape=(),dtype=tf.string, name='sentences2')
    text_input3 = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentences3')
    text_input4 = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentences4')
    text_input5 = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentences5')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs1 = preprocessing_layer(text_input1)
    encoder_inputs2 = preprocessing_layer(text_input2)
    encoder_inputs3 = preprocessing_layer(text_input3)
    encoder_inputs4 = preprocessing_layer(text_input4)
    encoder_inputs5 = preprocessing_layer(text_input5)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs1 = encoder(encoder_inputs1)
    outputs2 = encoder(encoder_inputs2)
    outputs3 = encoder(encoder_inputs3)
    outputs4 = encoder(encoder_inputs4)
    outputs5 = encoder(encoder_inputs5)
    attention1 = outputs1['pooled_output']
    attention2 = outputs2['pooled_output']
    attention3 = outputs3['pooled_output']
    attention4 = outputs4['pooled_output']
    attention5 = outputs5['pooled_output']
    attention_concentrate=tf.keras.layers.concatenate([attention1,attention2,attention3,attention4,attention5], axis=-1)
    attention_concentrate2 = keras.layers.Reshape(target_shape=(-1, 5, 512))(attention_concentrate)
    attention = tf.keras.layers.Dropout(0.1)(attention_concentrate)
    attention = tf.keras.layers.Dense(5, activation='tanh')(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(512)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    merge_model = tf.keras.layers.Multiply()([attention_concentrate2, attention])
    merge_model = tf.keras.layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(512,))(merge_model)
    merge_model= tf.keras.layers.Flatten(name='merge')(merge_model)
    output = tf.keras.layers.Dense(output_size, activation='softmax', name='output')(merge_model)

    return tf.keras.Model(inputs=[text_input1,text_input2,text_input3,text_input4,text_input5], outputs=output)

def bert_attention_for_text_2(tfhub_handle_preprocess, tfhub_handle_encoder, output_size):
    text_input1 = tf.keras.layers.Input(shape=(),dtype=tf.string, name='sentences1')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs1 = preprocessing_layer(text_input1)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs1 = encoder(encoder_inputs1)
    outputs1= keras.layers.Flatten(name='outputs1')(outputs1['pooled_output'])
    output = tf.keras.layers.Dense(output_size, activation='softmax', name='output')(outputs1)

    return tf.keras.Model(inputs=text_input1, outputs=output)

def save_train_data():
    music = models.Music.query.filter((models.Music.valid == True) & (models.Music.label != None)&(models.Music.bert_saved==True)).all()
    train_y=[]

    for i in music:
       full_signal, sr = librosa.load(i.audio_file_name)
       duration = 30
       sample_len = full_signal.shape[0]
       n_samples = int(sr * (duration + 10))
       lyric = i.normalized_lyric.split(' [seq] ')

       if sample_len <= n_samples :
            i.valid = False
            db.session.commit()
       else:
           file_name=generate_spectrogram(signal=full_signal[sr * 5: sr * (duration + 5)], sr=sr, n_fft=2048,
                                hop_length=1024, music=i, spectrogram_id=1)
           data = models.Text_train_data(text_data=' [seq] '.join(lyric[:8]), label=i.label, spectrogram_file =file_name)
           db.session.add(data)
           db.session.commit()
           train_y.append(i.label)
           if sample_len >= 2 * n_samples:
                file_name=generate_spectrogram(signal=full_signal[-sr * duration - 5:-5], sr=sr, n_fft=2048, hop_length=1024,
                                     music=i,
                                     spectrogram_id=2)
                data = models.Text_train_data(text_data=' [seq] '.join(lyric[-8:]), label=i.label, spectrogram_file=file_name)
                db.session.add(data)
                db.session.commit()
                train_y.append(i.label)
           if sample_len > 3 * n_samples:
                file_name=generate_spectrogram(signal=full_signal[sr * (duration + 4):sr * (duration * 2 + 4)], sr=sr, n_fft=2048,
                                     hop_length=1024,
                                     music=i,
                                     spectrogram_id=3)
                for j in range(11, 0, -1):
                    if j+8<=len(lyric):
                        data = models.Text_train_data(text_data=' [seq] '.join(lyric[j:j+8]), label=i.label, spectrogram_file =file_name)
                        db.session.add(data)
                        db.session.commit()
                        train_y.append(i.label)
                        break

    encoder = OneHotEncoder(handle_unknown='ignore')
    train_y = encoder.fit_transform(pd.DataFrame(train_y)).toarray()
    data=models.Text_train_data.query.all()
    for j in range(len(data)):
        data[j].one_hot_encoding=str(list(train_y[j]))
        db.session.commit()

def get_text_data():
    data = models.Text_train_data.query.all()
    ids=[]
    one_hot_encodings=[]
    for i in data:
        ids.append(i.id)
        one_hot_encodings.append(i.one_hot_encoding)
    return np.array(ids), np.array(one_hot_encodings)

def text_normalize(bert_model, bert_preprocess_model):
    music=models.Music.query.filter((models.Music.valid==True)&(models.Music.bert_saved==None)).all()
    for i in music:
        print (i.id)
        lyric=lower_case(i.lyric).split('\n')

        lyric.pop(0)
        lyric.pop(-1)
        lyric=[re.sub(r'\[.*\]', '', i) for i in lyric]
        lyric = [re.sub(r'\(.*\)', '', i) for i in lyric]
        lyric = [re.sub(r'\u2005', ' ', i) for i in lyric]
        lyric = [re.sub(r'\.', ' ', i) for i in lyric]
        lyric=[i for i in lyric if i!='']
        if len(lyric)<=7:
            i.valid=False
            db.session.commit()
        else:

            text_test = lyric
            try:
                i.normalized_lyric=' [seq] '.join(lyric)
                i.bert_saved=True
                db.session.commit()
            except Exception:
                i.valid=False
                db.session.commit()
def data_clean():
    music=models.Music.query.filter(models.Music.valid == True).all()
    for i in music:
        lyric = lower_case(i.lyric).split('\n')
        count=0
        if lower_case(i.music_name) not in lyric[0]:
            for j in lower_case(i.music_name).split(' '):
                if j in lyric[0]:
                    count+=1
            if count<=2:
                print(i.id)
                i.trash=True
                i.valid=False
                db.session.commit()
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
        db.session.remove()
        return tf.constant(batch_x, dtype=tf.string), np.asarray(batch_y).astype('float64')


if __name__ == "__main__":
    '''
    valid_data_check()
    
    data_clean()
    top_tag_select()
    text_normalize(bert_model, bert_preprocess_model)
    '''

    bert_model, bert_preprocess_model, tfhub_handle_preprocess, tfhub_handle_encoder = bert_models()


    
    save_train_data()
    

    '''
    one_hot_encoding=ast.literal_eval(models.Text_train_data.query.first().one_hot_encoding)

    classifier_model=bert_attention_for_text_2(tfhub_handle_preprocess, tfhub_handle_encoder, len(one_hot_encoding))
    classifier_model.save('./model2.h5')

    model2 = Model(classifier_model.input, classifier_model.get_layer('BERT_encoder').output)
    model2.save('./Bert_FEATURE_512.h5')

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    epochs = 5



    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=200,
                                              num_warmup_steps=20,
                                              optimizer_type='adamw')
    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)


    train_generator=My_Custom_Generator( 32)

    history = classifier_model.fit_generator(generator=train_generator, epochs=5, shuffle=True)
    '''


