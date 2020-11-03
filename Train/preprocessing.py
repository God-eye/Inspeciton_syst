from config import Config
import numpy as np
import os
import cv2
from tqdm import tqdm

############## Loading and preprocessing ##############
class Functions(Config):
	'''
	Docstring : This class will load the training or testing data and contains all the preprocessing functions
	'''
	def __init__(self):		
		Config.__init__(self, train_path, test_path, model_path, train_anom)
		self.load_buffer = {'frm_cnt': 0, 'indx':0, 'total':0}
		self.dr = 0
		# These are load buffers as the training videos are arranged in class wise directories.
		self.anom_dir_indx = {'Abuse': 0,
		                        'Arrest': 0,
		                        'Arson': 0,
		                        'Assault': 0,
		                        'Burglary': 0,
		                        'Explosion': 0,
		                        'Fighting': 0,
		                        'RoadAccidents': 0,
		                        'Robbery': 0,
		                        'Shooting': 0,
		                        'Shoplifting': 0,
		                        'Stealing': 0,
		                        'Vandalism': 0}

		self.anom_frm_cnt = {'Abuse': 0,
		                      'Arrest': 0,
		                      'Arson': 0,
		                      'Assault': 0,
		                      'Burglary': 0,
		                      'Explosion': 0,
		                      'Fighting': 0,
		                      'RoadAccidents': 0,
		                      'Robbery': 0,
		                      'Shooting': 0,
		                      'Shoplifting': 0,
		                      'Stealing': 0,
		                      'Vandalism': 0}

	def load_anom_batch(self):
	    classes = []
	    dirs = os.listdir(self.train_anom)

	    # We have 13 different classes
	    # For every class if a single video loaded then reset the buffer.
	    if self.dr >= 13:
	      self.dr = 0
	    dir = dirs[self.dr]
	    self.dr += 1
	    
	    path = os.path.join(self.train_anom, dir)
	    print(dir)
	    lst = sorted(os.listdir(path))

	    if self.anom_dir_indx[dir] >= len(lst):
	      return 0
	    
	    if not self.anom_frm_cnt[dir]:
	      self.anom_dir_indx[dir] += 1
	      self.load_buffer['total'] += 1

	    self.load_buffer['indx'] = self.anom_dir_indx[dir]
	    self.load_buffer['frm_cnt'] = self.anom_frm_cnt[dir]
	    vid = self.load_frames(os.path.join(path, lst[self.load_buffer['indx']]), agmt = False)
	    labels = dirs.index(dir)
	    self.anom_frm_cnt[dir] = self.load_buffer['frm_cnt']
	    self.anom_dir_indx[dir] = self.load_buffer['indx']
	    
	    '''except Exception as e:
	      print(e)
	      self.anom_frm_cnt[dir] = self.load_buffer['frm_cnt']
	      self.anom_dir_indx[dir] = self.load_buffer['indx']
	      continue'''

	    return vid, labels, self.load_buffer['total']

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
