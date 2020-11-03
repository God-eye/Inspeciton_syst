# -*- coding: utf-8 -*-
"""Model V2.0.ipynb
Written by : Aditya, Nikhil
"""
###################### Importing Libraries ###################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D,Conv2DTranspose, LayerNormalization, BatchNormalization, TimeDistributed, Conv2D, Flatten, Dense, Dropout
import keras
import concurrent.futures
import re
import pprint
from multiprocessing import Process

###################### copy the data in the gpu memory step by step ####################
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

###################### Parameter Initialization ########################################
class Config():
  def __init__(self, test_path, model_path, result_pth, train_path = None, img_size = (128, 128), batch_size = 8, mx_frm = 1600, stride = [1, 2], frm_cnt = 10, test_size = 400, epochs = 10, tst_seq = 300):
    self.train_path = train_path
    self.test_path = test_path
    self.img_size = img_size
    self.batch_size = batch_size
    self.model_path = model_path
    self.epochs = epochs
    self.result_pth = result_pth
    self.stride = stride
    self.mx_frm = mx_frm
    self.frm_cnt = frm_cnt
    self.test_size = test_size
    self.tst_seq = tst_seq

##################### Class Preprocessing Functions ###########################################
class Functions(Config):
  def __init__(self):
    Config.__init__(self, train_path, test_path, model_path,result_pth)

    # load buffer :- frm_cnt : stores the no of frames already loaded of the current video (None represents end of current video)
    #                indx : stores the indx of the video which is being processed / being loaded
    #                total : stores the amount of video loaded.                     
    self.load_buffer = {'frm_cnt': None, 'indx':0, 'total':0}

  def load_batch(self):
    '''
    DOCTYPE : This function will load the training videos in a batch of size defined in class Config.
    Input : None
    output : Batch of augmentd and processed clips, Total no of videos loaded
    '''
    clips = []
    a = 0
    q = 0
    for dir in tqdm(os.walk(train_path)):
      # os.walk() returns an array and the first element of the array represents the subdirectories in our main directory and we want to load the files in the subdirectories.
      # So we skip the first iteration.
      a += 1
      if a == 1:
        continue

      try:
        # If the frame count is None or zero then all the frames of that video are loaded and increment video index.
        if not self.load_buffer['frm_cnt']:
          self.load_buffer['indx'] += 1
          self.load_buffer['total'] += 1

        # Produced clips according to the load buffer indx. 
        pth = os.path.join(dir[0], sorted(dir[2])[self.load_buffer['indx']])
        clips.append(self.load_frames(pth))

      except Exception as e:
        print(e)
        # The training directory contains two folders so this step will start loading the videos from next directory.
        self.load_buffer['indx'] = 0
        continue

      break
    return clips, self.load_buffer['total']

  def load_frames(self, pth, agmt = True):
    '''
    DOCTYPE : This function will load a set of frame sequences from a given video.
    Input = pth - path of the video, agmt - True (Will apply augmentation) / False (will not apply augmentation)
    output = numpy array of frame sequences
    '''
    video = cv2.VideoCapture(pth)
    print('\n starting video no : ',self.load_buffer['total'])
    frames = []
    cnt = 0

    while video.isOpened:
      ret, frame = video.read()
      cnt += 1

      # If there is any error in loading the next frame. Might be because of ending of the video.
      
      if not ret:
        print('\nTotal frames read', cnt)
        self.load_buffer['frm_cnt'] = None
        print("\nvideo finished.")
        break

      # If frm_cnt exists then the previous video was not loaded completely and it will continue the previous sequence.
      if self.load_buffer['frm_cnt']:
        if self.load_buffer['frm_cnt'] <= cnt:
          img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          frame = cv2.resize(img/256, self.img_size)
        else:
          continue

      # If frm_cnt is None then it will start loading the videos from 1st frame.
      else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(img/256, self.img_size)
      #print('frame shape = ', frame.shape)
      frames.append(frame.reshape([self.img_size[0], self.img_size[1], 1]))

      # Specifies the maximum no of frames to be loaded
      if len(frames) >= self.mx_frm:
        break

    # update the frm_cnt variable according to whether the video is completed or not.
    if ret:
      self.load_buffer['frm_cnt'] = cnt
    else:
      self.load_buffer['frm_cnt'] = None
    
    video.release()

    # If the no of frames loaded are less than the the of frames specified in a sequence then it will dump that sequence.
    if len(frames) < self.frm_cnt:
      print('video has insufficiant frames')
      self.load_buffer['frm_cnt'] = None
      raise

    # Perform Augmentation
    if agmt:
      frames = self.augment(frames)
    
    return np.array(frames)

  def augment(self, frames):
    '''
    DOCTYPE : This function will Augment the frames according to the time series strides specified in the Config class.
    Input : Sequence of frames.
    Ouput : Augmented Sequence of frames.
    '''
    agmted = np.zeros((self.frm_cnt, self.img_size[0], self.img_size[1], 1))
    clips = []

    try:
      for strd in self.stride:
        for s in range(0, len(frames), strd):
          if len(frames[s:s+self.frm_cnt]) == 10:
            agmted[:,:,:,:] = frames[s:s+self.frm_cnt]
            clips. append(agmted)
    except:
      print('Error occured in augment')

    no = len(clips) % 8
    print("clips dropped ",no)
    clips = clips[:len(clips)-no]
    return clips

  def load_single_test(self):
    test = np.zeros((self.test_size, self.img_size[0], self.img_size[1], 1))
    for dir in os.listdir(self.test_path):
      path = os.path.join(self.test_path, dir)
      frames = self.load_frames(path, agmt = False)

    test = frames[0:self.test_size]
    del frames
    return test


####################### Model Architecture ##################################

class Model(Functions):
  def __init__(self):
    Functions.__init__(self)
    self.output1 = None
    self.output = None

  def anom(self):
    inputs = tf.keras.layers.Input(shape=[self.frm_cnt, self.img_size[0], self.img_size[1], 1])
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
    x = TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, self.frm_cnt, self.img_size[0], self.img_size[1], 1))(inputs)
    x = LayerNormalization()(x)
    for enc in encode:
      x = enc(x)
    self.output1 = x

    for dec in decode:
      x = dec(x)

    output = TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same"))(x)

    return tf.keras.Model(inputs=inputs, outputs = output)

  def spatial(self, filters, filter_size,stride , cnv = True, pading="same"):
    seq = tf.keras.Sequential()
    if cnv:
      seq.add(TimeDistributed(Conv2D(filters, filter_size, padding=pading)))
    else:
      seq.add(TimeDistributed(Conv2DTranspose(filters, filter_size, strides=stride, padding=pading)))
    seq.add(LayerNormalization())
    return seq

  def temporal(self, filters, filter_size, pading = "same", return_sequence=True):
    seq = tf.keras.Sequential()
    seq.add(ConvLSTM2D(filters, filter_size, padding=pading, return_sequences=return_sequence))
    seq.add(LayerNormalization())
    return seq

  def anom_type(self):
    seq = Sequential()
    seq.add(Flatten())
    seq.add(Dense(1000, activation='relu'))
    seq.add(Dropout(0.5))
    seq.add(Dense(512, activation='relu'))
    seq.add(Dropout(0.4))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.5))
    seq.add(Dense(13, activation='softmax'))
    return seq

def evaluate(test, typ):
    '''
    DOCTYPE : This function is used to returnn the result of anomaly detection algorithm.
    Input : A Video sequence to check
    Output : Write the prediction of the model in a txt file
    '''
    sz = test.shape[0] // 10
    sequences = np.zeros((sz, 10, img_dim[0], img_dim[1], 1))

    # apply the sliding window technique to get the sequences
    cnt = 0
    for i in range(0, test.shape[0], 10):
      if i + 10 <= test.shape[0]:
        sequences[cnt, :, :, :, :] = test[i:i+10]
        cnt += 1
    test = None
    clip = None
    
    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences,batch_size=4)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0,sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa
    
    while True:
      try:
        fle = open(cnfg.result_pth, 'w')
        break
      except:
        time.sleep(0.001)
        print('file is busy')
        continue
    flag = 0
    length = len(sr)
    ct = 0
    for i in sr:
      if i <= 0.96:
        ct += 1
      if (ct/length )== 0.3:
        flag = 1
    
    if flag:
      fle.write(typ)
      print('detected anomaly')    
    # if (sr<=0.96).any() or (sr<=0.96).all():
    #   fle.write(typ)
    #   print('detected anomaly')

    else:
      fle.write('Normal')
      print('Normal')
    
    fle.close()
    
    # #plot the regularity scores
    # print(sr)
    # plt.plot(sr)
    # plt.ylabel('regularity score Sr(t)')
    # plt.xlabel('frame t')
    # plt.show()
def play2(pth):
  time.sleep(7)
  vid = cv2.VideoCapture(pth)
  while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
      break
    frame = cv2.resize(frame,(512,512))
    cv2.imshow('vid', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
      break
  vid.release()
  cv2.destroyAllWindows()



def strt_eval(argmt):
  '''
  DOCTYPE : This function will start sequence processing
  '''
  frm = argmt[0]
  typ = argmt[1]
    
  frames = np.array(frm).reshape((cnfg.tst_seq, img_dim[0], img_dim[1], 1))
  evaluate(frames, typ)
  return 1

def test(test_path):
  '''
  DOCTYPE : Load the test video from test directory.
  Input : path of test Dir
  output : play the video in real time along with the analysis algorithm.
  '''
  for pth in os.listdir(test_path):
    tst_pth = os.path.join(test_path, pth)
    frames = []
    vid = cv2.VideoCapture(tst_pth)
    n = 0
    p0 = Process(target = play2, args = ([tst_pth]) )
    p0.start()
    
    while vid.isOpened():
      ret, frame = vid.read()
      if not ret:
        break
      n+= 1
      time.sleep(0.030)
      frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frm = cv2.resize(frm/256, img_dim)
      frames.append(frm.reshape((img_dim[0],img_dim[1], 1)))

      if n%cnfg.tst_seq == 0:
        print(n)
        temp = re.split(r'(\d+)', pth)[0]
        frames = np.array(frames).reshape((cnfg.tst_seq, img_dim[0], img_dim[1], 1))
        evaluate(frames,temp)
        # n = 0
        frames =[]
    p0.join()
    vid.release()
    cv2.destroyAllWindows()
  while True:
    try:
      fle = open(cnfg.resul_pth, 'w')
      break
    except:
      time.sleep(0.001)
      print('file is busy')
      continue
  fle.write('Video finished')
  fle.close()

if __name__ == '__main__':
  model_path = 'model_weights/anomaly_detect.h5'
  result_pth = 'IRIS_WEB/IRIS-backend/public/text_files/text.txt'
  test_path = 'Test'
  cnfg = Config(test_path, model_path,result_pth, tst_seq = 300)
  fncn = Functions()
  mdl = Model()
  img_dim = (128, 128)
  model = mdl.anom()
  model.compile(loss='mse',experimental_steps_per_execution = 50, optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
  try:
    model.load_weights('Model/tpu_model.h5')
    print('Model loaded successfuly')
  except:
    print("couldn't load the weights")
  # model = load_mdl()
  test(test_path)
  # test= fncn.load_single_test()
  # evaluate(test,'Abuse')
