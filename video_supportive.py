import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical

def readAndDivideData(sessions,monkey,array,seed,target_to_predict,
    lfp_for_denoising_start_ms,lfp_for_denoising_end_ms,
    sliceTimesToPredict_start_ms,sliceTimesToPredict_end_ms,lfp_ms_for_vel):
    dataFolder='../data_neural/'
    percent_outlier_exclude=0
    ds_ratio=1
    train_ratio=0.75
    val_ratio=0.15
    test_ratio=1-train_ratio-val_ratio
    zscore_by_trial=1

    if monkey=='Bx':
        rt_upperLim=600
        rt_lowerLim=200
    elif monkey =='Ls':
        rt_upperLim=400
        rt_lowerLim=0

    lfp_chosen='200to400Hz_envelope_MOVE'

    #read in data, combine sessions if needed
    for session in sessions:
        print(session)
        filepath = dataFolder+monkey+session+'MOVElfp_matrix_final-out'+str(percent_outlier_exclude) +'-ds_'+str(ds_ratio)+'_300to300'+lfp_chosen+'.mat'

        arrays = {}
        f = h5py.File(filepath,'r')
        for k, v in f.items():
          arrays[k] = np.array(v)

        if 'lfp_matrix' in locals():
            lfp_matrix=np.concatenate((lfp_matrix,arrays['lfp_matrix']),axis=0)
            tps_all_dirs=np.concatenate((tps_all_dirs,np.transpose(arrays['tp_kept'])), axis=0)#check axis
            pin_somatotopy_score,allKinVars_current=loadSomaAndKinVars(
                dataFolder,monkey,session,lfp_chosen)
            xVel_yVel_slices_current=loadKinSlices(dataFolder,monkey,session,
                '_velSlices_concise_selectedTrials_'+lfp_chosen)
            # kinProfileForPlotting_current=loadKinSlices(dataFolder,monkey,session,
            #     '_velSlices_concise_selectedTrials_'+lfp_chosen+'_forPlotting')
            allKinVars=mergeAllKinVars(allKinVars,allKinVars_current)
            xVel_yVel_slices=mergeKinSlices(xVel_yVel_slices,xVel_yVel_slices_current)
            #kinProfileForPlotting=mergeKinSlices(kinProfileForPlotting,kinProfileForPlotting_current)
        else:
            lfp_matrix=arrays['lfp_matrix']
            tps_all_dirs=np.transpose(arrays['tp_kept'])
            pin_somatotopy_score, allKinVars=loadSomaAndKinVars(
                dataFolder,monkey,session,lfp_chosen)
            xVel_yVel_slices=loadKinSlices(dataFolder,monkey,session,
                '_velSlices_concise_selectedTrials_'+lfp_chosen)
            # kinProfileForPlotting=loadKinSlices(dataFolder,monkey,session,
            #     '_velSlices_concise_selectedTrials_'+lfp_chosen+'_forPlotting')

    #choose time range and z score

    inital_z_score_start_ms=lfp_for_denoising_start_ms
    inital_z_score_end_ms=-400
    
    lfp_for_denoising_time=np.arange(lfp_for_denoising_start_ms,
        lfp_for_denoising_end_ms+0.5,0.5)
    zscore_reference_timeIdx_start=np.where(lfp_for_denoising_time==inital_z_score_start_ms)[0][0]
    zscore_reference_timeIdx_end=np.where(lfp_for_denoising_time==inital_z_score_end_ms)[0][0]


    lfp_time_ori=np.transpose(arrays['lfp_time'])[0]
    lfp_for_denoising_start_idx=np.where(lfp_time_ori==lfp_for_denoising_start_ms)[0][0]
    lfp_for_denoising_end_idx=np.where(lfp_time_ori==lfp_for_denoising_end_ms)[0][0]
    timeIdx_for_analysis=np.arange(
        lfp_for_denoising_start_idx,lfp_for_denoising_end_idx+1,1)
    lfp_time_for_analysis=lfp_time_ori[timeIdx_for_analysis]
    lfp_matrix_trimmed=lfp_matrix[:,timeIdx_for_analysis,:]


    if zscore_by_trial:
        lfp_matrix_trimmed_mean=np.mean(lfp_matrix_trimmed[:,zscore_reference_timeIdx_start:zscore_reference_timeIdx_end,:],
        axis=1,keepdims=True)
        lfp_matrix_trimmed_std=np.std(lfp_matrix_trimmed[:,zscore_reference_timeIdx_start:zscore_reference_timeIdx_end,:],
        axis=1,keepdims=True)#changed from mean... bug!
        lfp_matrix_trimmed_std[lfp_matrix_trimmed_std<0.00001]=1
        lfp_matrix_trimmed=(lfp_matrix_trimmed-lfp_matrix_trimmed_mean)/lfp_matrix_trimmed_std

    #choose trials
    kin_trial_filter=np.logical_and(allKinVars['RTrelative2max_ms']<rt_upperLim,
    allKinVars['RTrelative2max_ms']>rt_lowerLim)#logical_and can only combine two things


    trials_indices= np.where(np.logical_and(tps_all_dirs[:,0]<9,
    kin_trial_filter))#all directions

    tps=tps_all_dirs[trials_indices[0],:]
    tps_oneHot=convertLabelToOneHot(tps)

    
    print(tps.shape)
    #assert len(tps_oneHot) == len(np.unique(tps))
    print(tps_oneHot.shape)


    lfp_matrix_trimmed=lfp_matrix_trimmed[trials_indices[0],:,:]

    if target_to_predict=='vel':
        sliceTimes=np.linspace(-200,400,num=31)#steps of 20ms
        iSliceTimeToPredict_start_idx=np.where(sliceTimes==sliceTimesToPredict_start_ms)[0][0]
        iSliceTimeToPredict_end_idx=np.where(sliceTimes==sliceTimesToPredict_end_ms)[0][0]
        #31*#trial
        vel_final=np.vstack(((xVel_yVel_slices['xv_profile'][iSliceTimeToPredict_start_idx:iSliceTimeToPredict_end_idx+1,trials_indices[0]].T.flatten()),
            (xVel_yVel_slices['yv_profile'][iSliceTimeToPredict_start_idx:iSliceTimeToPredict_end_idx+1,trials_indices[0]].T.flatten())))
        vel_final=vel_final.T
        #check dim (2*_)   _ time 1 ~last in trial 1, time 1 ~last in trial 2,...    
        lfp_matrix_for_velPrediction=[]
        for iTrial in np.arange(lfp_matrix_trimmed.shape[0]):
            for iSliceTimeToPredict in np.arange(iSliceTimeToPredict_start_idx,iSliceTimeToPredict_end_idx+1):
                thisSliceTimeEnd_ms=sliceTimes[iSliceTimeToPredict]
                thisSliceTimeStart_ms=thisSliceTimeEnd_ms-lfp_ms_for_vel
                lfp_idx_start=np.where(lfp_time_for_analysis==thisSliceTimeStart_ms)[0][0]
                lfp_idx_end=np.where(lfp_time_for_analysis==thisSliceTimeEnd_ms)[0][0]
                lfp_matrix_for_velPrediction.append(np.squeeze(lfp_matrix_trimmed[iTrial,lfp_idx_start:lfp_idx_end+1,:]))

        lfp_matrix_for_velPrediction_array=np.asarray(lfp_matrix_for_velPrediction)
        print(lfp_matrix_for_velPrediction_array.shape)
        print(vel_final.shape)
        lfp_matrix_trimmed=lfp_matrix_for_velPrediction_array







    #rearrange in space
    pin_map = sio.loadmat(dataFolder+'pin_map_M1_'+monkey+'.mat')['pin_map']
    pin_map_current=np.int16(pin_map)
    pin_map_current=pin_map_current-1
    nRows=pin_map_current.shape[0] #16
    nCols=pin_map_current.shape[1] #8

    #trials*time*electrodes->trials*time*nRows*nCols*1
    lfp_matrix_trimmed_reshaped=lfp_matrix_trimmed.reshape((
        lfp_matrix_trimmed.shape[0],lfp_matrix_trimmed.shape[1],
        nRows,nCols,1))
    for iRow in np.arange(nRows):
        for iCol in np.arange(nCols):
            iElec=pin_map_current[iRow,iCol]
            lfp_matrix_trimmed_reshaped[:,:,iRow,iCol,0]=\
            lfp_matrix_trimmed[:,:,iElec]
    if array=='lower':
        lfp_matrix_final=lfp_matrix_trimmed_reshaped[:,:,8:16,:,:]
    elif array=='upper':
        lfp_matrix_final=lfp_matrix_trimmed_reshaped[:,:,0:8,:,:]
    elif array=='dual':
        lfp_matrix_final=lfp_matrix_trimmed_reshaped

    print(lfp_matrix_final.shape)

    if target_to_predict=='tp':
        X_train,X_val,X_test,y_train,y_val,y_test=divideXyIntoThreeSets(
            lfp_matrix_final,tps_oneHot,train_ratio,val_ratio,test_ratio,seed)
    elif target_to_predict=='vel':
        X_train,X_val,X_test,y_train,y_val,y_test=divideXyIntoThreeSets(
            lfp_matrix_final,vel_final,train_ratio,val_ratio,test_ratio,seed)

    print(X_train.shape)
    print(y_train.shape)


    return X_train,X_val,X_test,y_train,y_val,y_test


def divideXyIntoThreeSets(X,y,train_ratio,val_ratio,test_ratio,seed):
    X_train,X_else,y_train,y_else=train_test_split(X,y,
        test_size=1-train_ratio, shuffle=True,random_state=seed)
    X_val,X_test,y_val,y_test=train_test_split(X_else,y_else,
        test_size=test_ratio/(train_ratio+test_ratio), 
        shuffle=True,random_state=seed)
    return X_train,X_val,X_test,y_train,y_val,y_test

def convertLabelToOneHot(tps):
    unique_tps=np.sort(np.unique(tps))
    num_classes=len(unique_tps)
    tps_in_catIndex=tps
    #map each tp to a category
    mapping={}
    for iCategory in range(num_classes):
        mapping[unique_tps[iCategory]]=iCategory

    for i in range(len(tps)):
        tps_in_catIndex[i,0]=mapping[tps[i,0]]

    tps_oneHot=to_categorical(tps_in_catIndex, num_classes=num_classes)
    return tps_oneHot



def get_compositeR2_from_xv_yv(y_real,y_predicted):
    #samples*#vars
    print(y_real.shape)
    print(y_predicted.shape)
    numerator=np.sum((y_real-y_predicted)**2,axis=(0,1))
    denominator=np.sum((y_real-np.mean(y_real,axis=0))**2,axis=(0,1))
    r2=1-numerator/denominator
    return r2



def loadKinSlices(dataFolder,monkey,session,fileIdentifier):
    filepath = dataFolder+monkey+session+fileIdentifier+'.mat'
    kin_slices_wUseless=sio.loadmat(filepath)
    allKinNames=kin_slices_wUseless.keys()
    kin_slices=dict()
    for thisKinName in allKinNames:
        if thisKinName[0]!='_':
            kin_slices[thisKinName]=kin_slices_wUseless[thisKinName]#[0]
    return kin_slices



def loadSomaAndKinVars(dataFolder,monkey,session,lfp_chosen):
    filepath = dataFolder+'pin_somatotopy_score_'+monkey+'.mat'
    soma=sio.loadmat(filepath)
    pin_somatotopy_score=soma['pin_somatotopy_score']

    filepath = dataFolder+monkey+session+'_kinematicsVars_concise_selectedTrials_'+lfp_chosen+'.mat'
    allKinVars_wUseless=sio.loadmat(filepath)
    # 'insDelay_ms','RTrelative2max_ms','RTthreshold_ms','RTexitsCenter_ms',...
    #    'duration_ms','peakVel','peakVel_ms'

    allKinNames=allKinVars_wUseless.keys()

    allKinVars=dict()
    for thisKinName in allKinNames:
        if thisKinName[0]!='_':#header, version, ...
            allKinVars[thisKinName]=allKinVars_wUseless[thisKinName][0]

    return pin_somatotopy_score, allKinVars


def mergeAllKinVars(allKinVars,allKinVars_current):
    allKinNames=allKinVars.keys()
    for thisKinName in allKinNames:
        allKinVars[thisKinName]=np.concatenate((allKinVars[thisKinName],allKinVars_current[thisKinName]
        ),axis=0)#check axis
    return allKinVars



def mergeKinSlices(kinSlices,kinSlices_current):
    allKinNames=kinSlices.keys()
    for thisKinName in allKinNames:
        if 'profile' in thisKinName or 'traj' in thisKinName:
            kinSlices[thisKinName]=np.concatenate((kinSlices[thisKinName],kinSlices_current[thisKinName]
            ),axis=1)#check axis
    return kinSlices


