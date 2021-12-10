"""
A collection of models we'll use to attempt to classify videos.
"""
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
#from tensorflow.keras.layers.recurrent import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
#from locallyConnected3D import *
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length, image_shape,
                 saved_model=None, features_length=2048):
        """
        `model` = one of:
            lstm
            lrcn
            mlp
            conv_3d
            c3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()



        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)

            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.lrcn()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, 
                image_shape[0], image_shape[1], image_shape[2])
            self.model = self.conv_3d()
        elif model == 'conv_3d_2by2':
            print("Loading conv_3d_2by2")
            self.input_shape = (seq_length, 
                image_shape[0], image_shape[1], image_shape[2])
            self.model = self.conv_3d_2by2()
        elif model == 'conv_3d_cont':
            print("Loading conv_3d_cont")
            self.input_shape = (seq_length, 
                image_shape[0], image_shape[1], image_shape[2])
            self.model = self.conv_3d_cont()
        elif model == 'localConn':
            print("Loading localConn")
            self.input_shape = (seq_length, 
                image_shape[0], image_shape[1], image_shape[2])
            self.model = self.localConn()
        elif model == 'conv_3d_noSpace':
            print("Loading Conv3D no space")
            self.input_shape = (seq_length, 
                image_shape[0], image_shape[1], image_shape[2])
            self.model = self.conv_3d_noSpace()
        elif model == 'c3d':
            print("Loading C3D")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.c3d()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)

        if 'cont' not in model:#categorical
                    # Set the metrics. Only use top k if there's a need.
            metrics = ['accuracy']
            if self.nb_classes >= 10:
                metrics.append('top_k_categorical_accuracy')
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                               metrics=metrics)
        else:
            metrics = ['MeanSquaredError','MeanAbsoluteError']
            self.model.compile(loss='mean_squared_error', optimizer=optimizer,
                               metrics=metrics)#try MAE primary for now


        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        def add_default_block(model, kernel_filters, init, reg_lambda):

            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=L2_reg(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=L2_reg(l=reg_lambda))))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # max pool
            model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

            return model

        initialiser = 'glorot_uniform'
        reg_lambda  = 0.001

        model = Sequential()

        # first (non-default) block
        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                         kernel_initializer=initialiser, kernel_regularizer=L2_reg(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser, kernel_regularizer=L2_reg(l=reg_lambda))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # 2nd-5th (default) blocks
        model = add_default_block(model, 64,  init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def mlp(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def conv_3d_ori(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            32, (3,3,3), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model


    def localConn(self):
        model = Sequential()
        model.add(Conv3D(
            8, (10,1,1), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(10, 1, 1)))
        model.add(LocallyConnected3D(1, (2,2,2), activation='relu'))
        model.add(Conv3D(
            8, (10,1,1), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(5, 1, 1)))
        model.add(LocallyConnected3D(1, (2,2,2), activation='relu'))
        model.add(LocallyConnected3D(1, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(4, 1, 1), strides=(4, 1, 1)))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def conv_3d(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
            #changed for neural application
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            8, (10,1,1), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(10, 1, 1)))
        model.add(Conv3D(32, (10,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(5, 1, 1)))
        model.add(Conv3D(32, (2,2,2), activation='relu'))
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(4, 1, 1), strides=(4, 1, 1)))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model


    def conv_3d_cont(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
            #changed for neural application
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            8, (10,1,1), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(10, 1, 1)))
        model.add(Conv3D(32, (10,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(5, 1, 1)))
        model.add(Conv3D(32, (2,2,2), activation='relu'))
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        #model.add(MaxPooling3D(pool_size=(4, 1, 1), strides=(4, 1, 1)))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))#used to be none, everything below
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))

        return model



    def conv_3d_2by2(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
            #changed for neural application
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            8, (10,1,1), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(10, 1, 1)))
        model.add(Conv3D(64, (10,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(5, 1, 1), strides=(5, 1, 1)))
        model.add(Conv3D(64, (2,2,2), activation='relu'))
        model.add(Conv3D(64, (2,2,2), activation='relu'))
        #model.add(MaxPooling3D(pool_size=(4, 1, 1), strides=(4, 1, 1)))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model


    def conv_3d_noSpace(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
            #changed for neural application
            no space conv
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            8, (10,1,1), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(10, 1, 1)))
        model.add(Conv3D(64, (10,1,1), activation='relu'))#to 64?
        model.add(MaxPooling3D(pool_size=(10, 1, 1), strides=(5, 1, 1)))
        model.add(Conv3D(64, (2,1,1), activation='relu'))#to 64?
        model.add(Conv3D(16, (3,1,1), activation='relu'))
        model.add(MaxPooling3D(pool_size=(4, 1, 1), strides=(4, 1, 1)))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(Conv3D(256, (2,2,2), activation='relu'))
        # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model


        

    def c3d(self):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf

        With thanks:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
        """
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5b',
                         subsample=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
