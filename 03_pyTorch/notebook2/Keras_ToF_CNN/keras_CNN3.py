# %% Notes: https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
#The five steps in the deep learning life cycle are as follows: 
# (1) Define the model: model = ...s
# (2) Compile the model: model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# (3) Fit the model: history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
# (4) Evaluate the model: loss = model.evaluate(X, y, verbose=0)
# (5) Make predictions: yhat = model.predict(X)

# Machine learning model using Keras. General PROCEDURE:
# Once the model is created, you can config the model with losses and metrics 
# with model.compile(), train the model with model.fit(), or use the model to do 
# prediction with model.predict().

# Model definition
# (1) Input data
# (2) First 1D CNN layer
# (3) Second 1D CNN layer
# (4) Max pooling layer
# (5) Third and fourth 1D CNN layer
# (6) Average pooling layer
# (7) Dropout layer
# (8) Fully connected layer with Softmax activation

## Recipe to make predictions using keras model. https://www.projectpro.io/recipes/make-predictions-keras-model
# Recipe Objective:In ML, our main motive is to create a model that can predict the output from new data. We can do this by training the model.
# Step 1 - Import the library
#----------------- Training ----------------- 
# Step 2 - Loading the Dataset: (X_train, y_train), (X_test, y_test) = mnist.load_data()
# Step 3 - Creating model and adding layers: model = Sequential()
     # Step 3.1: add the layers by using 'add'
# Step 4 - Compiling the model: using optimiser
# Step 5 - Fitting the model:After fitting a model we want to evaluate the model. 
            # Here we  using model.evaluate to evaluate the model and it will give us the loss and the accuracy. Here we have also printed the score.
#----------------- Testing ----------------- 
# Step 6 - Evaluating the model: it will give us the loss and the accuracy
# Step 7 - Predicting the output

# %clear
# To use interactive plot 
# %matplotlib auto 


# sigmoid activation function: For small values (<-5), sigmoid returns a value close to zero, and for large values (>5) the result of the function gets close to 1.
# %% Import the required libraries
# --------------------------------Step 1----------------------------------
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

tStart=time.time()

# %%  load the dataset 
training = True

modelFile = "c"

matlabDat_train = loadmat('train_datFinalAbhishek_AllRec_Din_38_118.mat')
x_train = matlabDat_train['dat']
y_train = matlabDat_train['labels']
Nframes=matlabDat_train['Nframes'][0][0] # number of time-windows per waveform
winLen=matlabDat_train['winLen'][0][0] # number of samples per window
shift=matlabDat_train['shift'][0][0] # shifts between samples


# we can split the data in to test and train
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,shuffle=True,random_state=42)

# %% Define the CNN Model 
if training:
    # Data Preprocessing
    # --------------------------------Step 2----------------------------------
    # Loading the Dataset
    # Reshape input data into input format for training (and testing sets).
    x_train = np.expand_dims(x_train,axis=2)
    # Set the threshold for training model in lables
    y_train = (y_train>.25).astype("int")   
    y_train = np.expand_dims(y_train,axis=2)
    
    # --------------------------------Step 3----------------------------------
    # Initialize  CNN Model parameters
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
    y_train = np.delete(y_train,NegInds,axis=0)
    x_train = np.delete(x_train,NegInds,axis=0)

    ##    print('%i pos, %i neg (%i)\n'%(Nsamples_pos,np.size(np.squeeze(y_train))-Nsamples_pos,Nsamples_neg))
 
    # Creating model and adding layers
    # reshape data for keras: we need to get ready and transform our data. That means transforming our train and label sets into tensor data, since  works with it’s own type of data known as tensor(s)
    # Define sequential model with 'Nlayer'
    model = Sequential()
    inputSize=np.shape(x_train)[1]
    if (params['layerType']=='dense'):
        x_train=np.squeeze(x_train)
        y_train=np.squeeze(y_train)
    
    # Create 'Nlayer' number of Layers
    for l in range(params['Nlayer']):
        
        if (params['layerType']=='conv'):
            # we don't want conv kernels larger than the x
            if (params['kernelSize']>inputSize):
                break
        
            # add a Convolution1D layer
            model.add(Conv1D(params['Nfilts'],params['kernelSize'],
                     padding='same',
                     activation=params['activation'],
                     strides=2,
                     batch_input_shape=(None,inputSize,1)))
        else:
            # add a Fully connected dense layer
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
    
    # activation function
    model.add(Activation('sigmoid'))
    
    # --------------------------------Step 4---------------------------------- 
    # compile a model by using: optimizer, loss, metrics   
    # Compile the Model:Now that we got our tensors we use them along with the model in cnn function. There, we first set up the optimization algorithm and the loss function. We use the “adam” algorithm as optimizer and "binary_crossentropy" as loss function.
    # Initialize all parameters and compile our model with optimizer. for example adam, SGD, GradientDescent, Adagrad, Adadelta and Adamax
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc', tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(),])
    # --------------------------------Step 5---------------------------------- 
    # We can fit a model on the data we have and can use the model after that. While fitting we can pass various parameters like batch_size, epochs, verbose, validation_data and so on.
    # Fitting the model using the training data for fitting the model.
    # After compiling our model, we train our model by fit() method. Training and Testing the Nn: .
    # call the fit function on the keras model and send the x_train (features) and y_train (labels) sets. We also set the options for epochs.
    # Evaluating the Model: Verbosity mode. 0 = silent, 1 = progress bar.
    # train the model with a batch size of 'batch_size and a training and validation split of 80 to 20
    history = model.fit(x_train, y_train,
                      batch_size=params['batch_size'],
                      epochs=params['epochs'],
                      validation_split=0.2,
                      verbose=1)
    # model.summary() is used to see all parameters and shapes in each layers in our models.
    
    ## Plot the training-model parameters
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
# the training is over, it returns the result and the model is prepared to be used for generating predictions.
# %% Testing the model

model = load_model(modelFile)

testFile = "test_wax_2kg_dat.mat"
outputFile = testFile[0:-4]+".txt"

matlabDat_test = loadmat(testFile)
t = matlabDat_test['t'] # time
testCCE = matlabDat_test['dat']
testLabelsTrue = matlabDat_test['labels']

# we can split the data in to test and train
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,shuffle=True,random_state=42)

# calc the times corresponding to the centers of each frame
frameTimes = t
testLabels=np.zeros([frameTimes.size,testCCE.shape[1]])

fig = plt.figure(figsize=(15, 12))
plt.suptitle("CNN Testing", fontsize=18, y=0.95)
for setId in range(testCCE.shape[1]):
    data_test = np.squeeze(testCCE[:,setId])

    # zero pad data_test in start and end, so that we can have centered frames 
    # at the first and last time
    data_test = np.pad(data_test,(int(winLen/2),int(winLen/2)))

    # standardize the features
    data_test = data_test/(np.max(data_test))
##    data_test[np.isnan(data_test)]=0
    
    # split up test data and search for the arrival. Initialiase zero vector
    x_test = np.zeros([testCCE.shape[0],winLen])

    for i in range(np.shape(testCCE)[0]):
        x_test[i,:]=np.squeeze(data_test[i:i+winLen])

##    # standardize the features
##    x_test=(x_test-np.ones([np.shape(x_test)[0],1])*np.mean(x_test,axis=0))/(np.ones([np.shape(x_test)[0],1])*np.std(x_test,axis=0))
##    x_test[np.isnan(x_test)]=0
    x_test = np.expand_dims(x_test,axis=2)

    # remove zero padding
    data_test = data_test[int(winLen/2):-int(winLen/2)]
    # --------------------------------Step 6----------------------------------
    # Step 6 - Evaluating the model: After fitting a model we want to evaluate the model. Here we are using model.evaluate to evaluate the model and it will give us the loss and the accuracy. 
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    
    # --------------------------------Step 7----------------------------------
    # Finally we are predicting the output. For this we use another part of the data that we didnot use 
    # Predict the output (arrival time for this case), 
    testLabels[:,setId] = np.squeeze(model.predict(x_test))
    # print(testLabels)
    
    if (max(testLabels[:,setId])>0):
        testLabels[:,setId]=testLabels[:,setId]/max(testLabels[:,setId])
        
    ax = fig.add_subplot(5, 3, setId+1)  
    ax.plot(np.squeeze(frameTimes)*1e6,testLabels[:,setId]/max(testLabels[:,setId]), 'b', label='Test Labels')
    ax.plot(t*1e6,data_test/max(abs(data_test)), 'g', label='Data Test')
    ax.legend(loc='best'); ax.set_xlabel("Time [$\mu$s]");
    
    if np.shape(testLabelsTrue)[1]<2:
        ax.plot(t*1e6,testLabelsTrue/max(testLabelsTrue), 'r', label='Test Labels True')
        ax.legend(loc='best');ax.set_xlabel("Time [$\mu$s]");
    else:
        ax.plot(t*1e6,testLabelsTrue[:,setId]/max(testLabelsTrue[:,setId]), 'r', label='Test Labels True')
        ax.legend(loc='best'); ax.set_xlabel("Time [$\mu$s]");
            
    plt.ylim([0,1.5])
    
plt.tight_layout()


#precision = metrics.accuracy_score(y_pred, y_test) * 100
#print("Accuracy with SVM: {0:.2f}%".format(precision))


# export data for further post processing
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
# %% Find the maximum in an array and plot 
idx_tt=[0]*len(testLabels[1])
idx_tltnormax=[0]*len(testLabelsTrue[1])
relative_error = [0]*len(testLabels[1]) 
N =len(testLabels[1])

for i in range (N):
    tt = np.where(testLabels[:,i]==np.amax(testLabels[:,i]))
    tltnorm = testLabelsTrue[:,i]/max(testLabelsTrue[:,i])
    tltnormax = np.where(tltnorm==np.amax(tltnorm))
    idx_tt[i] =t[tt[0]];        # test labels
    idx_tltnormax[i]=t[tltnormax[0]];        # test labels true
    x = idx_tt[i]*1e6
    y = idx_tltnormax[i]*1e6
    relative_error[i] = np.abs(x - y)/x
    #print(idx_tt[i],idx_tltnormax[i])

plt.figure(figsize=(9, 6))
plt.subplot()
plt.plot(np.arange(N),np.array(idx_tltnormax)[:,0,0]*1e6,'*r', 
         markersize=12,label='test labels true')

plt.plot(np.arange(N),np.array(idx_tt)*1e6,'ok', 
         markersize=12,label='test labels predicted')
plt.xlabel('Data', fontsize=14, color='black')
plt.ylabel("Time [$\mu$s]", fontsize=14, color='black')
plt.legend(loc='upper left')
plt.show()
