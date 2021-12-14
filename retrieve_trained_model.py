import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
#from data import DataSet
from video_supportive import *
import time
import os.path
import os
import tensorflow as tf

from train import *
monkey='Bx'#'Bx'

if monkey=='Bx':
    allSessions=np.asarray(['171215','171220','171221',\
        '171128','171129','171130','171201b',\
        '180323','180322','180605'])
    allSessions_numDir=np.asarray([2,2,2,4,4,4,4,8,8,8])
elif monkey=='Ls':
    allSessions=np.asarray(['150930','151007','151014'])
    allSessions_numDir=np.asarray([8,8,8])

array='lower'#'lower'#upper#dual
#scrambleLocations=0


# it can be only 1 session as well, in a list []
target_to_predict='vel'#'tp'
model = 'conv_3d_cont'
#model = 'conv_3d_2by2'#'conv_3d_noSpace'#'conv_3d'#'lstm'
saved_model = None  # None or weights file
class_limit = None  # int, can be 1-101 or None

load_to_memory = True#False  # pre-load the sequences into memory
batch_size = 16#64#16#32#speed no difference?
nb_epoch = 500#1000
lfp_start_ms=-700
lfp_end_ms=1400#1400#1000

vel_start_ms=0
vel_end_ms=300
lfp_ms_for_vel=300#preceding ms until vel




if target_to_predict=='tp':
    seq_length = (lfp_end_ms-lfp_start_ms)*2+1
elif target_to_predict=='vel':
    seq_length = (lfp_ms_for_vel)*2+1



data_type = 'images'
if array=='dual':
    image_shape = (16, 8, 1)
else:
    image_shape = (8, 8, 1)
seed=8

nClasses=8
scrambleLocations=0
scrambleSeed=0


new_model = tf.keras.models.load_model('../data/models/Bx180323lowerbch16conv_3d_cont-training-_1639428502.3214514')

sessions=allSessions[allSessions_numDir==nClasses]
X_train,X_val,X_test,y_train,y_val,y_test=readAndDivideData(
    sessions,monkey,array,seed,target_to_predict,
    lfp_start_ms,lfp_end_ms,vel_start_ms,vel_end_ms,lfp_ms_for_vel,
    scrambleLocations,scrambleSeed)