
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

tStart=time.time()

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
   'layerType':['dense'],
   'Nfilts':[4],
   'kernel1Size':[5,10,15,20],
   'kernel2Size':[5,10,15,20],
   'kernel3Size':[5,10,15,20],
   'epochs':[4],
   'dropout':[0.2],
   'activation':['relu'],
   'batch_size':[100,150,200,250]}
   

def TOF_model(x_train,y_train,x_val,y_val,params):
    # reshape data for keras
    model = Sequential()
    inputSize=np.shape(x_train)[1]
    if (params['layerType']=='dense'):
        x_train=np.squeeze(x_train)
        x_val=np.squeeze(x_val)
        y_train=np.squeeze(y_train)
        y_val=np.squeeze(y_val)

    kernelSizes=[params['kernel1Size'],params['kernel2Size'],params['kernel3Size']]
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

# set the kernelSizes to zero for unused layers
for nl in range(1,3):
    metrics[(metrics[:,headers.index('Nlayer')]==nl),headers.index('kernel3Size')]=0
    if nl<2:
        metrics[(metrics[:,headers.index('Nlayer')]==nl),headers.index('kernel2Size')]=0
    
# write the metrics/headers to file
with open('HyperParamOpt_round3.txt','w') as f:
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

# remove all dependend variables except 1
headers_=copy.deepcopy(headers)

# remove the dep vars with only a single value
iter=0
while (iter<(np.shape(C)[0])):
    if (np.absolute(C[headers_.index('val_acc'),iter])<1e-5):
        for dir in range(2):
            C=np.delete(C,iter,dir)
        headers_.remove(headers_[iter])
    else:
        iter=iter+1

# plot correlation coefficients
plt.figure()
plt.imshow(C)
plt.xticks(range(np.size(headers_)),headers_,rotation=45,horizontalalignment="right")
plt.yticks(range(np.size(headers_)),headers_,rotation=45,horizontalalignment="right")
plt.colorbar()
[plt.text(i,j,'%.2e'%C[i,j],color='white',verticalalignment='center',horizontalalignment='center') for i in range(np.shape(C)[0]) for j in range(np.shape(C)[1])]
plt.tight_layout()

for key in remKeys:
    for dir in range(2):
        C=np.delete(C,headers_.index(key),dir)
    headers_.remove(key)


# find the top 2 correlations plot bar plots of val_acc vs var
corrOrd=np.argsort(np.absolute(C[headers_.index('val_acc'),:]))
Ncorr=2
plt.figure()
yvals=metrics_std[:,headers.index('val_acc')]
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
    plt.ylabel('standardized val_acc')
    [plt.text(valInd,errs[valInd][0],'%.2f'%errs[valInd][0],verticalalignment='bottom',horizontalalignment='center') for valInd in range(np.size(uxvals))]

plt.tight_layout()

var1='kernel1Size'
var2='kernel2Size'
C2=np.zeros([np.size(p[var1]),np.size(p[var2])])
# plot mean val_acc for each kernel1Size, kernel2Size combo
for k1 in range(np.size(p[var1])):
    for k2 in range(np.size(p[var2])):
        C2[k1,k2]=np.mean(metrics_std[(metrics[:,headers.index('Nlayer')]>=2)*(metrics[:,headers.index(var1)]==p[var1][k1])*(metrics[:,headers.index(var2)]==p[var2][k2]),headers.index('val_acc')])

plt.figure()
plt.imshow(C2)
[plt.text(i,j,'%.2e'%C2[i,j],color='red',verticalalignment='center',horizontalalignment='center') for i in range(np.shape(C2)[0]) for j in range(np.shape(C2)[1])]
plt.xticks(range(np.size(p[var2])),[str(p[var2][i]) for i in range(np.size(p[var2]))])
plt.yticks(range(np.size(p[var1])),[str(p[var1][i]) for i in range(np.size(p[var1]))])
plt.xlabel(var2)
plt.ylabel(var1)


plt.show()

