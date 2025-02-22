
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
p={'Nlayer':[1,2,3],
   'Nfilts':[4,8,16],
   'kernelSize':[10,25,50],
   'epochs':[2,3,4,5],
   'dropout':[0,0.2,0.5],
   'activation':['relu','elu'],
   'batch_size':[50]}
   

def TOF_model(x_train,y_train,x_val,y_val,params):
    # reshape data for keras
    model = Sequential()
    inputSize=np.shape(x_train)[1]
    for l in range(params['Nlayer']):
        # we don't want conv kernels larger than the x
        if (params['kernelSize']>inputSize):
            break
        
        # add a Convolution1D layer
        model.add(Conv1D(params['Nfilts'],params['kernelSize'],
                 padding='valid',
                 activation=params['activation'],
                 strides=1,
                 batch_input_shape=(None,inputSize,1)))
        
        # Measure size of last layer output to be input of next layer
        inputSize=model.layers[-1].output_shape[1]
        
    # add a dropout layer
    model.add(Dropout(params['dropout']))
    # add a dense layer
    model.add(Dense(1))
    model.add(MaxPooling1D(pool_size=inputSize))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc', talos.utils.metrics.f1score])
    
    history=model.fit(x_train, y_train,
                      validation_data=[x_val,y_val],
                      batch_size=params['batch_size'],
                      epochs=params['epochs'],
                      validation_split=.2,
                      verbose=0)
    return history, model


# run parameter sweep
scan_object=talos.Scan(x=x,
             y=y,
             model=TOF_model,
             params=p,
             experiment_name='TOF_cylinder',
             round_limit=50)

# retrieve the metrics
metrics=scan_object.data.values

# retrieve headers
headers=scan_object.data.columns

# assign values to string params
metrics[metrics=='relu']=1
metrics[metrics=='elu']=-1

metrics=metrics.astype('float')

# calc. correlation coeffs.
C=np.corrcoef(np.transpose(metrics))

# retrieve the corr coefs for the val_acc and val_loss rows
val_acc_ind=np.where(headers=='val_acc')
val_loss_ind=np.where(headers=='val_loss')
acc_ind=np.where(headers=='acc')
loss_ind=np.where(headers=='loss')
val_f1_ind=np.where(headers=='val_f1score')
f1_ind=np.where(headers=='f1score')


C_alt=C
C_alt[:,val_acc_ind]=0.
C_alt[:,vloss_ind]=0.
C_alt[:,acc_ind]=0.
C_alt[:,loss_ind]=0.
C_alt[:,val_f1_ind]=0.
C_alt[:,f1_ind]=0.

C_acc=np.squeeze(C_alt[val_acc_ind,:])
C_loss=np.squeeze(C_alt[val_loss_ind,:])
C_f1=np.squeeze(C_alt[val_f1_ind,:])


# plot corr coef bars
plt.figure()
plt.subplot(3,1,1)
plt.bar(range(1,np.size(C_acc)+1),C_acc)
plt.subplot(3,1,2)
plt.bar(range(1,np.size(C_acc)+1),C_loss)
plt.subplot(3,1,3)
plt.bar(range(1,np.size(C_acc)+1),C_f1)
plt.show()
