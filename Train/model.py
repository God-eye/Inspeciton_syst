########### Importing Libraries ##############
from preprocessing import Functions
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D,Conv2DTranspose, LayerNormalization, BatchNormalization, TimeDistributed, Conv2D, Flatten, Dense, Dropout
import keras
import pprint

########### Model ##################

class Model(Functions):
  def __init__(self):
    Functions.__init__(self)
    self.output1 = None
    self.output = None

  def anom(self, algo2 = False):
  	'''
  	Docstring : Spatial and Temporal series based anomaly detection algorithm 
  	'''
    inputs = tf.keras.layers.Input(shape=[10, self.img_size[0], self.img_size[1], 1])
    encode = [
              self.spatial(64, (5,5), stride = 2, pading="same", cnv=True),
              self.temporal(64, (3,3), pading='same'),
              self.temporal(32, (3,3), pading='same')
    ]
    decode = [
              self.temporal(64, (3,3), pading='same'),
              self.spatial(64,(5,5), stride = 2, pading="same", cnv = False),
              self.spatial(128, (11,11), stride= 2, pading="same", cnv= False)
    ]
    seq = tf.keras.Sequential()
    x = TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, self.img_size[0], self.img_size[1], 1))(inputs)
    x = LayerNormalization()(x)
    for enc in encode:
      x = enc(x)
    self.output1 = x
    print(x.shape)
    if algo2:
      return self.output1

    for dec in decode:
      x = dec(x)

    output = TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same"))(x)

    return tf.keras.Model(inputs=inputs, outputs = output)

  def spatial(self, filters, filter_size,stride , cnv = True, pading="same"):
  	'''
  	Docstring : Spatial Encoding
  	'''
    seq = tf.keras.Sequential()
    if cnv:
      seq.add(TimeDistributed(Conv2D(filters, filter_size, padding=pading)))
    else:
      seq.add(TimeDistributed(Conv2DTranspose(filters, filter_size, strides=stride, padding=pading)))
    seq.add(LayerNormalization())
    return seq

  def temporal(self, filters, filter_size, pading = "same", return_sequence=True):
  	'''
  	Docstring : Temporal Encoding
  	'''
    seq = tf.keras.Sequential()
    seq.add(ConvLSTM2D(filters, filter_size, padding=pading, return_sequences=return_sequence))
    seq.add(LayerNormalization())
    return seq

  def anom_class(self):
  	'''
  	Docstring : Video classification model using 3d convolutional
  	'''
    inputs = tf.keras.layers.Input(shape=[self.frm_cnt, self.img_size[0], self.img_size[1], 1])
    x = tf.keras.layers.Conv3D(128, (3,3,3), activation='relu', input_shape = (self.frm_cnt, self.img_size[0], self.img_size[1], 1))(inputs)
    x = tf.keras.layers.MaxPool3D((2,2,2))(x)
    x = LayerNormalization()(x)
    x = tf.keras.layers.Conv3D(32, (3,3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPool3D((2,2,2))(x)
    x = LayerNormalization()(x)
    x = tf.keras.layers.Conv3D(8, (3,3,3), activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(13, activation='softmax')(x)
    return tf.keras.Model(inputs = inputs, outputs=outputs)