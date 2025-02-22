# NOTE: Some time the "val_false_positives" name doent match, so check
# "history.history" to confirm the names
# Update: 10/12/2023 Added clear session
# %% 
# ------------------- Step 1: Import the required libraries ------------------- 
# Import the library
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import load_model
##from keras import backend as bknd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn import metrics
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import SeparableConv1D

import h5py
import mat73
# release the global state.
tf.keras.backend.clear_session()
tStart=time.time()

# %%  load large dataset using h5py format.
numRun = 1  # for running multiple test
for ModRun in range(numRun):    
    # dataFileName  = 'training_data_Rec_7to11_Din_random_80%_11012024_test1.mat'
    dataFileName  = 'training_data_Rec_7to11_Din_random_1%_11012024_test' + str(ModRun+1) +'.mat'
    matlabDat_train = h5py.File(dataFileName, 'r')
    print(matlabDat_train.keys()) 
    
    filepath = "D:\\CNN_John\\keras_CNN\\c_" + dataFileName[0:-4];
    # filepath = "C:\\Users\\376189\\Documents\\CNN_TOF\\c" + dataFileName[0:-4]+"_run" + str(ModRun+1);
    #modelFile = "c_" + dataFileName[0:-4] #+"_run" +str(ModRun+1); # generate model for testing the unknown data
    
    #print(matlabDat_train['X'][:]) 
    
    x_train = np.transpose(matlabDat_train['dat'][:])
    y_train = np.transpose(matlabDat_train['labels'][:])
    Nframes=matlabDat_train['Nframes'][0][0] # number of time-windows per waveform
    winLen=matlabDat_train['winLen'][0][0] # number of samples per window
    shift=matlabDat_train['shift'][0][0] # shifts between samples
    
    
    # %% Define the CNN Model 
    # ------------------- Step 2: Data Preprocessing--------------------------♦
    # Loading the Dataset
    x_train = np.expand_dims(x_train,axis=2);       # Training Data
    y_train = (y_train>.25).astype("int");          # Training Labels, set threshold
    y_train = np.expand_dims(y_train,axis=2);
    
    # we can split the data in to test and train
    #X_train, X_test, y_train, y_test = 
    #train_test_split(X,y,test_size=0.20,shuffle=True,random_state=42)
    
    # ------------------- Step 3 Define CNN Model ------------------- 
    params={'Nlayer':5,
       'layerType':'conv',
       'Nfilts':16,
       'kernelSize':50, # filter size for convolution
       'epochs':15,
       'dropout':0.2,
       'activation':'relu',
       'batch_size':250,
       'pos_weight':1.,
       'class_sample_ratio':10.,
       'pool_size':2}          # For maxpooling}
    
    # Linear (sequential) stack of layers  
    model = Sequential()
    
    # reduce the number of negative-label samples
    Nsamples_pos = np.sum(np.squeeze(y_train))
    Nsamples_neg_in = np.size(np.squeeze(y_train))-Nsamples_pos
    Nsamples_neg = np.min([int(params['class_sample_ratio']*Nsamples_pos),Nsamples_neg_in])
    
    # find inds of neg samples
    NegInds = np.squeeze(np.where(np.squeeze(y_train<.5)))
    
    # randomly select samples to remove
    rng = np.random.default_rng()
    selections = rng.choice(np.size(NegInds),size=(Nsamples_neg_in-Nsamples_neg),replace=False)
    NegInds = [NegInds[s] for s in selections]
    
    # prune negative-labels
    x_train = np.delete(x_train,NegInds,axis=0)   # x_train is the training data set.
    y_train = np.delete(y_train,NegInds,axis=0)   # y_train is the set of labels to all the data in x_train. 

    ## print('%i pos, %i neg (%i)\n'%(Nsamples_pos,np.size(np.squeeze(y_train))-Nsamples_pos,Nsamples_neg))
     
    model = Sequential()
    inputSize=np.shape(x_train)[1]
    if (params['layerType']=='dense'):
        x_train=np.squeeze(x_train)
        y_train=np.squeeze(y_train)
    
    # # kernelSize = params['kernelSize']
    # # Create 'Nlayer' number of Layers
    # for l in range(params['Nlayer']):  
    #     if (params['layerType']=='conv'):
    #         # we don't want conv kernels larger than the x
    #         if (params['kernelSize']>inputSize):
    #             # kernelSize = inputSize
    #             break
    #         model.add(Conv1D(params['Nfilts'],params['kernelSize'],
    #                  padding='same',
    #                  activation=params['activation'],
    #                  strides=2,
    #                  batch_input_shape=(None,inputSize,1)))
    #     else:
    #         # add a Fully connected dense layer
    #         model.add(Dense(int(kernelSizes[l]),input_shape=(inputSize,)))
    #         model.add(Activation(params['activation']))                      
        
    #     # Measure size of last layer output to be input of next layer
    #     inputSize=model.layers[-1].output_shape[1]
        
    # model.add(Dropout(params['dropout']))
    # model.add(Dense(1))
    
    # if (params['layerType']=='conv'):
    #     model.add(MaxPooling1D(pool_size=inputSize))
    
    # model.add(Activation('sigmoid'))
    
    # # ------------------- Step 4 -------------------  
    # # compile a model by using: optimizer, loss, metrics   
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['acc', tf.keras.metrics.FalsePositives(),
    #                        tf.keras.metrics.FalseNegatives(),])
    
    # # ------------------- Step 5 ------------------- 
    # # Fit a model on the data we have and can use the model after that. 
    
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, 
    #                              save_best_only=True,mode='min')
    
    # history = model.fit(x_train, y_train,
    #                   batch_size=params['batch_size'],
    #                   epochs=params['epochs'],
    #                   validation_split=0.2,
    #                   callbacks=[checkpoint],
    #                   verbose=1)
    
    # # save the trained model, if not using check pointing
    # # model.save(modelFile)
    
    # tStop=time.time()
    kernelSize = params['kernelSize']
    # Create 'Nlayer' number of Layers
    for l in range(params['Nlayer']):  
        if (params['layerType']=='conv'):
            if (params['kernelSize']>inputSize):
                kernelSize = inputSize
                #print('kernelSize is greater than inputSize, So no additional conv layer',l)
                #break
            print('Adding Conv Layer',l)
            # model.add(Conv1D(params['Nfilts']*2**l,params['kernelSize'],
            #          padding='same',
            #          activation=params['activation'],
            #          strides=1, # Change the conv1d strides to 1 since we’re going to include maxpool layers afterwards
            #          batch_input_shape=(None,inputSize,1)))
            model.add(SeparableConv1D(params['Nfilts']*2**l,
                     params['kernelSize'],
                     padding='same',
                     activation=params['activation'],
                     strides=1, # Change the conv1d strides to 1 since we’re going to include maxpool layers afterwards
                     batch_input_shape=(None,inputSize,1)))
            #  include a pooling layer, and possibly either a batch norm or layer norm layer
            model.add(tf.keras.layers.Normalization())
            model.add(tf.keras.layers.MaxPooling1D(params['pool_size']))
            
        # Measure size of last layer output to be input of next layer
        inputSize = model.layers[-1].output_shape[1]
        
    model.add(tf.keras.layers.Flatten())
    model.add(Dropout(params['dropout']))
    model.add(Dense(1))
    #model.add(MaxPooling1D(pool_size=inputSize))
    model.add(Activation('sigmoid'))
      
    # compile a model by using: optimizer, loss, metrics   
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc', tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.FalseNegatives(),])
    
    # Fit a model on the data we have and can use the model after that. 
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, 
                                 save_best_only=True, mode='min')
    
    history = model.fit(x_train, y_train,
                      batch_size=params['batch_size'],
                      epochs=params['epochs'],
                      validation_split=0.2,
                      callbacks=[checkpoint],
                      verbose=0)
    tStop=time.time()
    print('Training time: %.1f s\n'%(tStop-tStart))
    
    # %% Plot the training-model parameters
    plt.figure(1)
    plt.subplot(4,1,1)
    plt.plot(history.history['val_acc'])
    plt.title('val accuracy')
    plt.subplot(4,1,2)
    plt.plot(history.history['val_loss'])
    plt.title('val loss')
    plt.subplot(4,1,3)
    plt.plot(history.history['val_false_positives'])
    plt.title('false positives')
    plt.subplot(4,1,4)
    plt.plot(history.history['val_false_positives'])
    plt.title('false negatives')
    plt.tight_layout()
    plt.savefig(filepath+"\\Training.png")

    # clear and delete the model
    # K.clear_session()
    # del model

# %% summarize the model
model.summary()
# %%  Visualize Filters
# summarize filter shapes
for layer in model.layers:
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)
    
    
for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    # get filter weights
    filters= layer.get_weights()
    print(filters.shape)
