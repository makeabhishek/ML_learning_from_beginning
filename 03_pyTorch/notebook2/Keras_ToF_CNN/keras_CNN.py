
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
##from keras import backend as bknd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

outputFile="labels.txt"

##x=loadmat('fakeToFdata.mat')
x=loadmat('test_dat.mat')
data=x['dat']
labels=x['labels']
Nframes=x['Nframes'][0][0] # number of time-windows per waveform
winLen=x['winLen'][0][0] # number of samples per window
shift=x['shift'][0][0] # shifts between samples
##t=x['t'] # time
##data_test=x['testDat']

x_test=loadmat('testWaves_TOFchirp_Al_ID4.5in_OD5in_alt.mat')
t=x_test['t'] # time
testCCE=x_test['testData']
testLabelsTrue=x_test['testLabels']

data=np.expand_dims(data,axis=2)
labels=(labels>.25).astype(int)
labels=np.expand_dims(labels,axis=2)

filters=8
batch_size=100
kernel_size=25
epochs=3

model = Sequential()
# we add a Convolution1D, which will learn filters
# group filters of size filter_length:
##model.add(Conv1D(filters,
##                 kernel_size,
##                 padding='valid',
##                 activation='relu',
##                 strides=1,
##                 batch_input_shape=(None,data.shape[1],1)))

##model.add(Activation('relu'))
model.add(Dense(10,activation='relu'))

# we use max pooling:
model.add(MaxPooling1D(pool_size=2,strides=2))

model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 batch_input_shape=(None,int(data.shape[1]/2),1)))

##model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=2,strides=2))

model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 batch_input_shape=(None,26,1)))

# we use max pooling:
model.add(MaxPooling1D(pool_size=2,strides=2))

model.add(Dropout(0.2))
##model.add(MaxPooling1D(pool_size=2,strides=3))

model.add(Dense(1))
model.add(MaxPooling1D(pool_size=3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
##model.summary()

model.fit(data, labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=.2)


# calc the times corresponding to the centers of each frame
frameTimes=t[int(winLen/2):-1-int(winLen/2):shift]

testLabels=np.zeros([Nframes,testCCE.shape[1]])
plt.figure()
for setId in range(testCCE.shape[1]):
    data_test=np.squeeze(testCCE[:,setId])
    # split up test data and search for the arrival
    testFrames=np.zeros([Nframes,winLen])
    
    for i in range(Nframes):
        testFrames[i,:]=np.squeeze(data_test[i*shift:i*shift+winLen])
    testFrames=np.expand_dims(testFrames,axis=2)

    # predict the arrival time
    testLabels[:,setId]=np.squeeze(model.predict(testFrames))

    
    plt.subplot(5,2,setId+1)
    plt.plot(t,data_test/max(abs(data_test)),frameTimes,testLabels[:,setId]/max(testLabels[:,setId]),frameTimes,testLabelsTrue[setId,:]/max(testLabelsTrue[setId,:]))
    

file=open(outputFile,"w")
file.write("time [us],")
for i in range(testCCE.shape[1]):
    file.write(" label %i," % i)

file.write("\n")
for i in range(testLabels.shape[0]):
    file.write("%d, " % frameTimes[i])
    for j in range(testLabels.shape[1]):
        file.write("%d" % testLabels[i,j])
        if j+1==testLabels.shape[1]:
            file.write('\n')
        else:
            file.write(', ')

file.close()
plt.show()
