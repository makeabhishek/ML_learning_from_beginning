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

tStart=time.time()

training=False
testFile="test_canTOFchirp_30Aug19.mat"
testFile="test_COMSOL_waterTOF_variousID.mat"
outputFile=testFile[0:-4]+".txt"
modelFile="trainedCNN_opt"

matlabDat_train=loadmat('train_dat.mat')
x_train=matlabDat_train['dat']
y_train=matlabDat_train['labels']
Nframes=matlabDat_train['Nframes'][0][0] # number of time-windows per waveform
winLen=matlabDat_train['winLen'][0][0] # number of samples per window
shift=matlabDat_train['shift'][0][0] # shifts between samples

if training:
    
    x_train=np.expand_dims(x_train,axis=2)
    y_train=(y_train>.25).astype(int)
    y_train=np.expand_dims(y_train,axis=2)

    params={'Nlayer':3,
       'layerType':'conv',
       'Nfilts':16,
       'kernelSize':50,
       'epochs':6,
       'dropout':0.2,
       'activation':'relu',
       'batch_size':250,
       'pos_weight':1.,
       'class_sample_ratio':10.}


    model = Sequential()

    # reduce the number of negative-label samples
    Nsamples_pos=np.sum(np.squeeze(y_train))
    Nsamples_neg_in=np.size(np.squeeze(y_train))-Nsamples_pos
    Nsamples_neg=np.min([int(params['class_sample_ratio']*Nsamples_pos),Nsamples_neg_in])

    # find inds of neg samples
    NegInds=np.squeeze(np.where(np.squeeze(y_train<.5)))

    # randomly select samples to remove
    rng=np.random.default_rng()
    selections=rng.choice(np.size(NegInds),size=(Nsamples_neg_in-Nsamples_neg),replace=False)
    NegInds=[NegInds[s] for s in selections]

    # prune negative-labels
    y_train=np.delete(y_train,NegInds,axis=0)
    x_train=np.delete(x_train,NegInds,axis=0)

    ##    print('%i pos, %i neg (%i)\n'%(Nsamples_pos,np.size(np.squeeze(y_train))-Nsamples_pos,Nsamples_neg))

    # reshape data for keras
    model = Sequential()
    inputSize=np.shape(x_train)[1]
    if (params['layerType']=='dense'):
        x_train=np.squeeze(x_train)
        y_train=np.squeeze(y_train)

    for l in range(params['Nlayer']):
        
        if (params['layerType']=='conv'):
            # we don't want conv kernels larger than the x
            if (params['kernelSize']>inputSize):
                break
        
            # add a Convolution1D layer
            model.add(Conv1D(params['Nfilts'],params['kernelSize'],
                     padding='valid',
                     activation=params['activation'],
                     strides=1,
                     batch_input_shape=(None,inputSize,1)))
        else:
            # add a dense layer
            model.add(Dense(int(kernelSizes[l]),input_shape=(inputSize,)))
            model.add(Activation(params['activation']))                      
        
        # Measure size of last layer output to be input of next layer
        inputSize=model.layers[-1].output_shape[1]
        
    # add a dropout layer
    model.add(Dropout(params['dropout']))
    # add a dense layer
    model.add(Dense(1))

    if (params['layerType']=='conv'):
        model.add(MaxPooling1D(pool_size=inputSize))
        
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc', tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(),])
    history=model.fit(x_train, y_train,
                      batch_size=params['batch_size'],
                      epochs=params['epochs'],
                      validation_split=.2,
                      verbose=0)
    ##
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
    plt.plot(history.history['val_false_negatives'])
    plt.title('false negatives')
    plt.tight_layout()

    tStop=time.time()
    ##
    # save the trained model
    model.save(modelFile)
    print('Training time: %.1f s\n'%(tStop-tStart))

model = load_model(modelFile)

matlabDat_test=loadmat(testFile)
t=matlabDat_test['t'] # time
testCCE=matlabDat_test['dat']
testLabelsTrue=matlabDat_test['labels']



# calc the times corresponding to the centers of each frame
frameTimes=t
testLabels=np.zeros([frameTimes.size,testCCE.shape[1]])
plt.figure(2)
for setId in range(testCCE.shape[1]):
    data_test=np.squeeze(testCCE[:,setId])

    # zero pad data_test so that we can have centered frames at the first and last time
    data_test=np.pad(data_test,(int(winLen/2),int(winLen/2)))

    # standardize the features
    data_test=data_test/(np.max(data_test))
##    data_test[np.isnan(data_test)]=0
    
    # split up test data and search for the arrival
    x_test=np.zeros([testCCE.shape[0],winLen])
    
    for i in range(np.shape(testCCE)[0]):
        x_test[i,:]=np.squeeze(data_test[i:i+winLen])

##    # standardize the features
##    x_test=(x_test-np.ones([np.shape(x_test)[0],1])*np.mean(x_test,axis=0))/(np.ones([np.shape(x_test)[0],1])*np.std(x_test,axis=0))
##    x_test[np.isnan(x_test)]=0
    x_test=np.expand_dims(x_test,axis=2)

    

    # remove zero padding
    data_test=data_test[int(winLen/2):-int(winLen/2)]

    # predict the arrival time
    testLabels[:,setId]=np.squeeze(model.predict(x_test))
    if (max(testLabels[:,setId])>0):
        testLabels[:,setId]=testLabels[:,setId]/max(testLabels[:,setId])

    plt.subplot(5,np.ceil(np.shape(testLabels)[1]/5),setId+1)
    plt.plot(np.squeeze(frameTimes)*1e6,testLabels[:,setId]/max(testLabels[:,setId]))
    plt.plot(t*1e6,data_test/max(abs(data_test)))
    if np.shape(testLabelsTrue)[1]<2:
        plt.plot(t*1e6,testLabelsTrue/max(testLabelsTrue))
    else:
        plt.plot(t*1e6,testLabelsTrue[:,setId]/max(testLabelsTrue[:,setId]))
            
    plt.ylim([0,1])
plt.tight_layout()

file=open(outputFile,"w")
file.write("time [us],")
for i in range(testCCE.shape[1]):
    file.write(" label %i," % i)

file.write("\n")
for i in range(testLabels.shape[0]):
    file.write("%f, " % frameTimes[i])
    for j in range(testLabels.shape[1]):
        file.write("%f" % testLabels[i,j])
        if j+1==testLabels.shape[1]:
            file.write('\n')
        else:
            file.write(', ')

file.close()
plt.show()
