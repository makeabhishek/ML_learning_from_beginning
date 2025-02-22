
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

##testFile="test_canTOFchirp_30Aug19.mat"
testFile="COMSOL_waterTOF_variousID.mat"
outputFile="COMSOL_waterTOF_variousID_labels.txt"
modelFile="trainedCNN"

##x=loadmat('fakeToFdata.mat')
x=loadmat('train_dat.mat')
data=x['dat']
labels=x['labels']
Nframes=x['Nframes'][0][0] # number of time-windows per waveform
winLen=x['winLen'][0][0] # number of samples per window
shift=x['shift'][0][0] # shifts between samples

# load a saved model
model = load_model(modelFile) 

x_test=loadmat(testFile)
t=x_test['t'] # time
testCCE=x_test['dat']
testLabelsTrue=x_test['labels']

# calc the times corresponding to the centers of each frame
##frameTimes=t[int(winLen/2)-1:-1-int(winLen/2)+1]
frameTimes=t
testLabels=np.zeros([frameTimes.size,testCCE.shape[1]])
plt.figure(2)
for setId in range(testCCE.shape[1]):
    data_test=np.squeeze(testCCE[:,setId])

    # zero pad data_test so that we can have centered frames at the first and last time
    data_test=np.pad(data_test,(int(winLen/2),int(winLen/2)))
    
    # split up test data and search for the arrival
    testFrames=np.zeros([testCCE.shape[0],winLen])
    
    for i in range(np.shape(testCCE)[0]):
        testFrames[i,:]=np.squeeze(data_test[i:i+winLen])
    testFrames=np.expand_dims(testFrames,axis=2)

    # remove zero padding
    data_test=data_test[int(winLen/2):-int(winLen/2)]

    # predict the arrival time
    testLabels[:,setId]=np.squeeze(model.predict(testFrames))
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
