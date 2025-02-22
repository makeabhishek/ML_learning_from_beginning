
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
   'layerType':['dense','conv'],
   'Nfilts':[4,8,16],
   'kernelSize':[25,49,81],
   'epochs':[4],
   'dropout':[0.2],
   'activation':['relu'],
   'batch_size':[50]}
   

def TOF_model(x_train,y_train,x_val,y_val,params):
    # reshape data for keras
    model = Sequential()
    inputSize=np.shape(x_train)[1]
    if (params['layerType']=='dense'):
        x_train=np.squeeze(x_train)
        x_val=np.squeeze(x_val)
        y_train=np.squeeze(y_train)
        y_val=np.squeeze(y_val)
        
    for l in range(params['Nlayer']):
        # we don't want conv kernels larger than the x
        if (params['kernelSize']>inputSize):
            break
        if (params['layerType']=='conv'):
            # add a Convolution1D layer
            model.add(Conv1D(params['Nfilts'],params['kernelSize'],
                     padding='valid',
                     activation=params['activation'],
                     strides=1,
                     batch_input_shape=(None,inputSize,1)))
        else:
            # add a dense layer
            model.add(Dense(int(params['kernelSize']**.5),input_shape=(inputSize,)))
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
             round_limit=20)

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
with open('HyperParamOpt_round2.txt','w') as f:
	for i in range(np.size(headers)):
		f.write(str(headers[i])+', ')
	f.write('\r\n')
	np.savetxt(f,metrics,fmt='%.5e',delimiter=', ',newline='\r\n')

# standardize metrics
metrics_std=(metrics-np.ones([np.shape(metrics)[0],1],'float')*np.mean(metrics,axis=0))/(np.ones([np.shape(metrics)[0],1],'float')*np.std(metrics,axis=0))
                             
# calc. correlation coeffs.
C=np.corrcoef(np.transpose(metrics_std))
C[np.isnan(C)]=0

# retrieve the corr coefs for the val_acc and val_loss rows
inds={'val_acc': np.where(headers=='val_acc'),
      'val_loss': np.where(headers=='val_loss'),
      'acc': np.where(headers=='acc'),
      'loss': np.where(headers=='loss'),
      'val_f1score': np.where(headers=='val_f1score'),
      'f1score': np.where(headers=='f1score')}


# define keys of dependent variables
remKeys=['val_loss','val_f1score','acc','loss','f1score']

    
headers_=copy.deepcopy(headers)
for key in remKeys:
    for dir in range(2):
        C=np.delete(C,headers_.index(key),dir)
    headers_.remove(key)

# plot correlation coefficients
plt.figure()
plt.imshow(C)
plt.xticks(range(np.size(headers_)),headers_,rotation=45,horizontalalignment="right")
plt.yticks(range(np.size(headers_)),headers_,rotation=45,horizontalalignment="right")
plt.tight_layout()


##C_vals=[]
##plotKeys=['val_acc','val_loss','val_f1score']
##plt.figure()
##for key in plotKeys:
##    val=np.squeeze(C[inds[key],:])
##    lab=copy.deepcopy(headers)
##    for remKey in remKeys:
##        val=np.delete(val,inds[remKey])
##        lab.remove(remKey)
##    C_vals.append([val,lab])
##    plt.subplot(1,len(plotKeys),plotKeys.index(key)+1)
##    plt.bar(range(1,np.size(val)+1),val)
##    plt.xticks(np.linspace(1.,np.size(val)+1.,np.size(val)),lab,rotation=45,horizontalalignment="right")
##    plt.ylabel(key)
##    
##plt.tight_layout()
##plt.subplots_adjust(bottom=.25)

# find the top 2 correlations plot bar plots of val_acc vs var
corrOrd=np.argsort(C[headers_.index('val_acc'),:])
Ncorr=2
plt.figure()
yvals=metrics[:,headers.index('val_acc')]
for varInd in range(Ncorr):
    xvals=metrics[:,headers.index(headers_[corrOrd[-2-varInd]])]

    uxvals=np.unique(xvals)
    plt.subplot(1,Ncorr,varInd+1)
    errs=np.zeros([np.size(uxvals),2])
    for valInd in range(np.size(uxvals)):
        errs[valInd,:]=[np.mean(yvals[xvals==uxvals[valInd]]),np.std(yvals[xvals==uxvals[valInd]])]
        
    plt.bar(range(np.size(uxvals)),errs[:,0],yerr=errs[:,1])
    plt.xlabel(headers_[corrOrd[-2-varInd]])
    plt.xticks(range(np.size(uxvals)),uxvals.astype(str))
    plt.ylabel('val_acc')

plt.tight_layout()
plt.show()

