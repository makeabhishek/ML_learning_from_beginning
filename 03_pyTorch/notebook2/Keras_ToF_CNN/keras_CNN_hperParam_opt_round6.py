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
import sys
import talos
import copy
import time
import csv
import tensorflow as tf



tStart=time.time()
metricSaveFile='HyperParamOpt_round6.txt'

##testFile="test_canTOFchirp_30Aug19.mat"
testFile="COMSOL_waterTOF_variousID.mat"
outputFile="labels_2kg.txt"
modelFile="trainedCNN"

##x=loadmat('fakeToFdata.mat')
data=loadmat('train_dat_norm.mat')
x=data['dat']
y=data['labels']
Nframes=data['Nframes'][0][0] # number of time-windows per waveform
winLen=data['winLen'][0][0] # number of samples per window
shift=data['shift'][0][0] # shifts between samples


x=np.expand_dims(x,axis=2)
y=(y>.25).astype(int)
y=np.expand_dims(y,axis=2)

##x=talos.utils.rescale_meanzero(x)

# define hyperparams.
p={'Nlayer':[3,4,5],
   'layerType':['conv'],
   'Nfilts':[8],
   'kernelSize':[40,45,50,55,60],
   'epochs':[4,5,6],
   'dropout':[0.2],
   'activation':['relu'],
   'batch_size':[250],
   'pos_weight':[1.],
   'class_sample_ratio':[1.,2.,5.,10.]}
   
numConfigs=1
for par in p:
    numConfigs=numConfigs*np.size(p[par])
    
def TOF_model(x_train,y_train,x_val,y_val,params):
    # define sample weights
##    sample_weights=np.squeeze(y_train.astype(float)*params['pos_weight']+(1-y_train).astype(float))

    # reduce the number of negative-label samples
    Nsamples_pos=np.sum(np.squeeze(y_train))
    Nsamples_neg_in=np.size(np.squeeze(y_train))-Nsamples_pos
    Nsamples_neg=np.min([int(params['class_sample_ratio']*Nsamples_pos),Nsamples_neg_in])
##    print('\n %i, %i\n'%(int(params['class_sample_ratio']*Nsamples_pos),Nsamples_neg_in))   
##    print('%i pos, %i neg, factor = %.1f\n'%(Nsamples_pos,Nsamples_neg_in,params['class_sample_ratio']))

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
        x_val=np.squeeze(x_val)
        y_train=np.squeeze(y_train)
        y_val=np.squeeze(y_val)

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
                      validation_data=[x_val,y_val],
                      batch_size=params['batch_size'],
                      epochs=params['epochs'],
                      validation_split=.2,
                      verbose=0)
    return history, model

# limit the number of configurations to observe
numConfigs=np.min([60,numConfigs])

# run parameter sweep
scan_object=talos.Scan(x=x,
             y=y,
             model=TOF_model,
             params=p,
             experiment_name='TOF_cylinder',
             round_limit=60)

tStop=time.time()
dt=tStop-tStart
dt_hr=np.floor(dt/3600.)
dt_min=np.floor(dt/60.)-dt_hr*60
dt_s=np.floor(dt)-dt_hr*3600-dt_min*60
print('Elapsed time: %i hr %i min %i s'%(dt_hr,dt_min,dt_s))
# retrieve the metrics
metrics=scan_object.data.values

# retrieve headers
headers=scan_object.data.columns.tolist()

# assign values to string params
metrics[metrics=='relu']=1
metrics[metrics=='elu']=-1

metrics[metrics=='conv']=1
metrics[metrics=='dense']=-1

metrics=metrics.astype('float')

# write the metrics/headers to file
with open(metricSaveFile,'w') as f:
	for i in range(np.size(headers)):
		f.write(str(headers[i])+', ')
	f.write('\n')
	np.savetxt(f,metrics,fmt='%.5e',delimiter=', ',newline='\n')



