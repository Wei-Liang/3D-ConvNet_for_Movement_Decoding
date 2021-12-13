"""
Train our RNN on extracted features or images.
"""
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
#from data import DataSet
from video_supportive import *
import time
import os.path
import os

def train(data_type, seq_length, model, nClasses, sessions, 
    monkey, array, seed, target_to_predict,
    lfp_start_ms,lfp_end_ms,vel_start_ms,vel_end_ms,lfp_ms_for_vel,
    scrambleLocations,scrambleSeed,
    saved_model=None,
    class_limit=None, image_shape=None,
    load_to_memory=False, batch_size=32, nb_epoch=100):

    # Helper: Save the model. For generator at the moment
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('..','data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('..','data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: Save results.
    timestamp = time.time()
    if len(sessions)>1:
        csv_logger_path = os.path.join('..','data', 'logs', monkey+
            str(nClasses)+'dir'+array+'bch'+str(batch_size)+
            model + '-' + 'training-' )
    else:
        csv_logger_path = os.path.join('..','data', 'logs', monkey+
            sessions[0]+array+'bch'+str(batch_size)+
            model + '-' + 'training-')
    if scrambleLocations==1:
        csv_logger_path=csv_logger_path+'scramble'+str(scrambledSeed)

    csv_logger=CSVLogger(csv_logger_path+'_'+str(timestamp) + '.log')

    # # Get the data and process it.
    # if image_shape is None:
    #     data = DataSet(
    #         seq_length=seq_length,
    #         class_limit=class_limit
    #     )
    # else:
    #     data = DataSet(
    #         seq_length=seq_length,
    #         class_limit=class_limit,
    #         image_shape=image_shape
    #     )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    #steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X_train,X_val,X_test,y_train,y_val,y_test=readAndDivideData(
            sessions,monkey,array,seed,target_to_predict,
            lfp_start_ms,lfp_end_ms,vel_start_ms,vel_end_ms,lfp_ms_for_vel,
            scrambleLocations,scrambleSeed)
        print(X_train.shape)
        print(y_train.shape)
        # X, y = data.get_all_sequences_in_memory('train', data_type)
        # X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators. # not modified
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(nClasses, model, seq_length, 
        image_shape, saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)

        #save model
        modelPath=os.path.join('..','data', 'models', monkey+
            sessions[0]+array+'bch'+str(batch_size)+
            model + '-' + 'training-')
        if scrambleLocations==1:
            modelPath=modelPath+'scramble'+str(scrambledSeed)
        modelPath=modelPath+str(timestamp)
        os.mkdir(modelPath)
        rm.model.save(modelPath)

        #loss and metrics on test set
        print('onTestSet:')
        print(rm.model.metrics_names)
        scores= rm.model.evaluate(X_test, y_test, verbose=2)
        print(scores)

        #R2s for train, val and test
        y_predicted_train=rm.model.predict(X_train)
        y_predicted_val=rm.model.predict(X_val)
        y_predicted_test=rm.model.predict(X_test)
        r2_from_xv_yv_train=get_compositeR2_from_xv_yv(y_train,y_predicted_train)
        r2_from_xv_yv_val=get_compositeR2_from_xv_yv(y_val,y_predicted_val)
        r2_from_xv_yv_test=get_compositeR2_from_xv_yv(y_test,y_predicted_test)
        print('R2 train '+ str(r2_from_xv_yv_train) + ', val ' + 
            str(r2_from_xv_yv_val) + ', test '+str(r2_from_xv_yv_test))


    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)

def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    
    monkey='Bx'

    if monkey=='Bx':
        allSessions=np.asarray(['171215','171220','171221',\
            '171128','171129','171130','171201b',\
            '180323','180322','180605'])
        allSessions_numDir=np.asarray([2,2,2,4,4,4,4,8,8,8])
    elif monkey=='Ls':
        allSessions=np.asarray(['150930','151007','151014'])
        allSessions_numDir=np.asarray([8,8,8])

    array='lower'#'lower'#upper#dual

    nClasses=4
    #sessions=allSessions[allSessions_numDir==nClasses]
    sessions=['171130']
    # it can be only 1 session as well, in a list []

    model = 'localConn'#'conv_3d_noSpace'#'conv_3d_noSpace'#'conv_3d'#'lstm'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    
    load_to_memory = True#False  # pre-load the sequences into memory
    batch_size = 16#32
    nb_epoch = 500#1000
    lfp_start_ms=-700
    lfp_end_ms=1000
    seq_length = (lfp_end_ms-lfp_start_ms)*2+1


    data_type = 'images'
    if array=='dual':
        image_shape = (16, 8, 1)
    else:
        image_shape = (8, 8, 1)
    seed=8

    # Chose images or features and image shape based on network.
    # if model in ['conv_3d', 'c3d', 'lrcn']:
    #     data_type = 'images'
    #     image_shape = (80, 80, 3)
    # elif model in ['lstm', 'mlp']:
    #     data_type = 'features'
    #     image_shape = None
    # else:
    #     raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, nClasses, sessions, monkey, array, seed, 
        lfp_start_ms,lfp_end_ms,saved_model=saved_model,
        class_limit=class_limit, image_shape=image_shape,
        load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
