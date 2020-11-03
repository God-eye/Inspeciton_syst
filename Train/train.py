########################## Import Libraries ##########################
from config import Config
from preprocessing import Functions
from model import Model
import numpy as np
from tqdm import tqdm

############# Uncomment the section below if you are using tpu ##################
tpu = False

# import pprint

# if 'COLAB_TPU_ADDR' not in os.environ:
#   print('ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!')
# else:
#   tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
#   print ('TPU address is', tpu_address)

#   with tf.compat.v1.Session(tpu_address) as session:
#     devices = session.list_devices()
#   tpu = True
#   print('TPU devices:')
#   pprint.pprint(devices)

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# # This is the TPU initialization code that has to be at the beginning.
# tf.tpu.experimental.initialize_tpu_system(resolver)
# print("All devices: ", tf.config.list_logical_devices('TPU'))
# strategy = tf.distribute.TPUStrategy(resolver)

############################ Training Section ##################################
def evaluate(tst, labels):
    """
    Docstring : This function is used to find the anomaly part of the clip and train our classifier on just the anomaly part of the video and ignore the rest of the video.
    """
    sz = tst.shape[0] // 10
    sequences = np.zeros((sz, 10, cnfg.img_size[0], cnfg.img_size[1], 1))

    # apply the sliding window technique to get the sequences
    cnt = 0
    for i in range(0, tst.shape[0], 10):
      if i + 10 <= tst.shape[0]:
        sequences[cnt, :, :, :, :] = tst[i:i+10]
        cnt += 1

    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences,batch_size=8)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0,sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa
    
    flag = 0
    cnt = 0
    for i in sr:
      if i <= 0.965:
        cnt += 1
      if cnt / len(sr) >= 0.35:
        flag = 1
        break
        
    if flag:
      print('Anomaly Detected')
      batch = np.array(augment(tst))
      lbls = np.zeros((len(batch), 13))
      for i in range(len(batch)):
        lbls[i, :] = labels
      del tst
      seq1.fit(batch, lbls, batch_size=cnfg.batch_size, epochs=cnfg.epochs, shuffle=False)
      seq1.save_weights('../model_weights/classifier.h5', overwrite=True)
     # !cp "./model_weights/classifier.h5" -r "/content/drive/My Drive/Model V2.0 weights/"
      del batch

    else:
      print('everything is normal')
    #plot the regularity scores
    #plt.plot(sr)
    #plt.ylabel('regularity score Sr(t)')
    #plt.xlabel('frame t')
    #plt.show()


model_path = '../model_weights'
train_path = 'train_data/normal'
test_path = '../Test'
train_anom = 'train_data/anomaly'
cnfg = Config(train_path, test_path, model_path, train_anom)
fncn = Functions()
mdl = Model()


if tpu:
	# only 1 model can be loaded in the tpu at a time so we chose anomaly detection algorithm. If you want you can change that to classificaiton model.
	with strategy.scope():
		model = mdl.anom()
		model.compile(loss='mse', metrics = ['accuracy'], experimental_steps_per_execution = 50, optimizer=keras.optimizers.Adam(lr=1e-5, decay=1e-5, epsilon=1e-6))
	seq1 = mdl.anom_class()
	seq1.compile(loss='categorical_crossentropy', metrics = ['accuracy'], experimental_steps_per_execution = 50, optimizer=keras.optimizers.Adam(lr=1e-5, decay=1e-5, epsilon=1e-6))
else:
	model = mdl.anom()
	model.compile(loss='mse', metrics = ['accuracy'], experimental_steps_per_execution = 50, optimizer=keras.optimizers.Adam(lr=1e-5, decay=1e-5, epsilon=1e-6))
	seq1 = mdl.anom_class()
	seq1.compile(loss='categorical_crossentropy', metrics = ['accuracy'], experimental_steps_per_execution = 50, optimizer=keras.optimizers.Adam(lr=1e-5, decay=1e-5, epsilon=1e-6))


############# Anomaly detecting Training #######################
while True:
    clips, ttl = fncn.load_batch()
    if ttl == 800:
      clips = None
      break
    #seq1.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
    seq1.fit(clips, clips, batch_size=cnfg.batch_size, epochs=cnfg.epochs, shuffle=False)
    seq1.save_weights('../model_weights/anomaly_detect.h5', overwrite=True)

"""
Note : Train Each model at a time. Its not end to end training.
Step 1: Train anomaly detection network.
Step 2: Comment out the above Section and uncomment the section below.
Step 3: Run the script again to train video classification model.
"""

############# Classification model Training ####################
# while True:
#   clips, labels, total = fncn.load_anom_batch()
#   print('clips shape = ', clips[0].shape)
  
#   # apply the sliding window technique to get the sequences
#   lbl = np.zeros((1,13)) 
#   lbl[0][labels] = 1
#   for i in tqdm(range(0, clips.shape[0], 40)):
#     if i + 40 <= clips.shape[0]:
#       evaluate(clips[i:i+40], lbl)
  
#   del clips
#   del labels