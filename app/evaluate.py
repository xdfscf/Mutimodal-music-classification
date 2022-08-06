import ast
import copy
import datetime
import random
from xmlrpc.client import DateTime

import keras
import librosa
import requests
from matplotlib import pyplot as plt
from PIL import Image
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
from keras.callbacks import ModelCheckpoint
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
def get_combine_data(block_number,type):
    if type == "train":
        data = models.Text_train_data.query.filter(
            (models.Text_train_data.block != block_number) | (models.Text_train_data.block == None)).all()
    elif type == "validate":
        data = models.Text_train_data.query.filter_by(block=block_number).all()
    elif type=="all":
        data = models.Text_train_data.query.all()
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
def get_image_data(block_number,type):
    if type == "train":
        data = models.Text_train_data.query.filter(
            (models.Text_train_data.block != block_number) | (models.Text_train_data.block == None)).all()
    elif type == "validate":
        data = models.Text_train_data.query.filter_by(block=block_number).all()
    elif type=="all":
        data = models.Text_train_data.query.all()

    batch_y = []
    images = []
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
def get_text_data(block_number,type):
    if type == "train":
        data = models.Text_train_data.query.filter(
            (models.Text_train_data.block != block_number) | (models.Text_train_data.block == None)).all()
    elif type == "validate":
        data = models.Text_train_data.query.filter_by(block=block_number).all()
    elif type=="all":
        data = models.Text_train_data.query.all()
    batch_x = ['  '.join(i.text_data.split(' [SEP] ')) for i in data]
    batch_y = [ast.literal_eval(i.one_hot_encoding) for i in data]

    db.session.remove()

    return tf.constant(batch_x, dtype=tf.string), np.asarray(batch_y).astype('float64')
import gc


def evaluate_image_model():
    model_list = ["validate0-2-improvement-50-0.94.hdf5", "validate1-improvement-245-0.95.hdf5"
                  ,"validate2-improvement-248-0.95.hdf5","validate3-improvement-249-0.98.hdf5",
                  "validate4-improvement-243-0.86.hdf5"]
    count=0
    for i in model_list:
        model = keras.models.load_model(i, custom_objects={'KerasLayer': hub.KerasLayer, "K": K})
        train_x, train_y = get_image_data(count, "train")
        model.evaluate(train_x, train_y)
        train_x, train_y = get_image_data(count, "validate")
        model.evaluate(train_x, train_y)
        del train_x
        del train_y
        gc.collect()
        count+=1
        print(i)

def evaluate_text_model():
    model_list = ["./final_bert0.hdf5","./final_bert1.hdf5",
                  "./final_bert2.hdf5","./final_bert3.hdf5",
                  "./final_bert4.hdf5"]
    count = 0
    for i in model_list:
        model = keras.models.load_model(i, custom_objects={'KerasLayer': hub.KerasLayer, "K": K})
        train_x, train_y = get_text_data(count, "train")
        model.evaluate(train_x, train_y)
        train_x, train_y = get_text_data(count, "validate")
        model.evaluate(train_x, train_y)

        count += 1
        print(i)
def evaluate_combine_model():
    model_list=["./final_combine0.hdf5","./final_combine1.hdf5",
                "./final_combine2.hdf5","./final_combine3.hdf5",
                "./final_combine4.hdf5"]
    count = 0
    for i in model_list:
        model = keras.models.load_model(i, custom_objects={'KerasLayer': hub.KerasLayer, "K": K})
        train_x, train_y = get_combine_data(count, "train")
        model.evaluate(train_x, train_y)
        train_x, train_y = get_combine_data(count, "validate")
        model.evaluate(train_x, train_y)
        del train_x
        del train_y
        gc.collect()
        count += 1
        print(i)

evaluate_combine_model()