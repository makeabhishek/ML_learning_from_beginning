# For testing multiple trainign data and saving multiple data, 
# Date:1/21/2024

# Import the library
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
##from keras import backend as bknd
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import math
import tensorflow as tf
import time
import csv
import re
import numpy as np
import scipy.io

# release the global state.
tf.keras.backend.clear_session()
tStart=time.time()

# %% Reading csv file
numFiles = 1; # to run multiple test in random files

for jj in range (numFiles):
    csvFilenames = []
    diametermix = []
    #testDatFileName = 'filename_testData_random_80%_11012024_test1.csv'
    testDatFileName = 'filename_testData_random_99%_18012024_test'  + str(jj+1) + '.csv'
    modelFile = "c_training_data_Rec_7to11_Din_random_99%_11012024_test" + str(jj+1)
    
    with open(testDatFileName, mode ='r')as csvFile:  
      # reading the CSV file♦
      csvReader = csv.reader(csvFile)
      for row in csvReader:
          csvFilenames.append(row[0])
          #print(lines)
    
    testFileNo = len(csvFilenames)
    diameter= [[] * testFileNo for i in range(numFiles)]
    rmse_error = [[] * testFileNo for i in range(numFiles)]
    ToFtrue =np.zeros(shape=(15, testFileNo))
    ToFpredict = np.zeros(shape=(15, testFileNo))

    diametermix = []
    #modelFile = "c_training_data_Rec_7to11_Din_random_80%_11012024_test1"
    #modelFile = "c_training_data_Rec_7to11_Din_random_90%_10242023_test" + str(jj+1)
    model = load_model(modelFile)
    
    for ii in range(testFileNo):
        testFile = csvFilenames[ii]
        diametermix.append(testFile[28:35])
        diameter[jj].append(re.findall("\d+\.\d+", diametermix[ii]))
        print(testFile)
                
        # %% Testing the model
        outputFile = testFile[0:-4]+".txt";
        matlabDat_test = loadmat(testFile)
        
        # %%
        winLen = matlabDat_test['winLen'][0][0] # number of samples per window
        timSmpl = 500; # Number of times sample used for testing the data
        t = matlabDat_test['t'][0:timSmpl,:] # time
        testCCE = matlabDat_test['dat'][0:timSmpl,:];                # unknown test data
        testLabelsTrue = matlabDat_test['labels'][0:timSmpl,:];      # True values for validation
        
        frameTimes = t
        testLabels=np.zeros([frameTimes.size,testCCE.shape[1]])
        
        # fig = plt.figure(figsize=(15, 12))
        # plt.suptitle("CNN Testing", fontsize=18, y=0.95)
        for setId in range(testCCE.shape[1]):
            data_test = np.squeeze(testCCE[:,setId])
            data_test = np.pad(data_test,(int(winLen/2),int(winLen/2)))
            data_test = data_test/(np.max(data_test))    
            x_test = np.zeros([testCCE.shape[0],winLen])
        
            for i in range(np.shape(testCCE)[0]):
                x_test[i,:]=np.squeeze(data_test[i:i+winLen])
        
            x_test = np.expand_dims(x_test,axis=2)
            data_test = data_test[int(winLen/2):-int(winLen/2)] 
            testLabels[:,setId] = np.squeeze(model.predict(x_test))
    
            
            if (max(testLabels[:,setId])>0):
                testLabels[:,setId]=testLabels[:,setId]/max(testLabels[:,setId])
        
        # %% Find the maximum in an array and plot 
        idx_tt=[0]*len(testLabels[1])
        idx_tltnormax=[0]*len(testLabelsTrue[1])
        relative_error = [0]*len(testLabels[1]) 
        N =len(testLabels[1])
        
        for i in range (N):
            # select the time samples you want to consider to find the max. 
            tt = np.where(testLabels[:,i]==np.amax(testLabels[0:timSmpl,i]))
            tt= np.array(tt)
            if len(tt[0,:])>0:
                tt = np.take(tt, tt.size // 2)
                
            tltnorm = testLabelsTrue[:,i]/max(testLabelsTrue[:,i]); # tlt= test label true
            tltnormax = np.where(tltnorm==np.amax(tltnorm))
            idx_tltnormax[i] = t[tltnormax[0]];     # test labels true
            
            idx_tt[i] = t[tt];                      # test labels t[tt[0]];
                
            x = idx_tt[i]*1e6
            y = idx_tltnormax[i]*1e6
            relative_error[i] = np.abs(x - y)/x
        
        idx_tt= np.array(idx_tt)
        
        y_predicted= np.array(idx_tt)*1e6;           
        y_actual  = np.array(idx_tltnormax)[:,0,0]*1e6; 
        # y_actual[5:10]; y_predicted[5:10] # for selected receivers
        # ToFtrue[jj].extend(y_actual)
        # ToFpredict[jj].append(y_predicted)
        ToFtrue[:,ii] =  y_actual[:]
        ToFpredict[:,ii] = y_predicted[:,0]
        # [idx].extend    
        # %% Relative error
        relative_error = [0]*len(testLabels[1]) # np.empty(N)
        y_actual  = y_actual.reshape((15, 1))
        relative_error = np.abs(y_actual - y_predicted ) / y_actual
        
        MSE = np.square(np.subtract(y_actual[5:10],y_predicted[5:10])).mean()
        RMSE = math.sqrt(MSE)
        print("Root Mean Square Error:\t",'%.3f'%(RMSE))
        rmse_error[jj].append(RMSE)
        
    saveMatFile = 'testingResult_'+ modelFile[30:]+".mat"
    savemat(saveMatFile, {"ToFtrue": ToFtrue, "ToFpredict": ToFpredict, "csvFilenames": csvFilenames})
        
    # if (jj>0):
    #     diameter_new = diameter.copy()
    #     diameterAll = list(zip(diameterAll, diameter_new))
    #     rmse_error_new = rmse_error.copy() 
    #     rmse_errorAll = list(zip(rmse_errorAll, rmse_error_new))
        
            
# rmse_error_array = np.array(rmse_error)
# diameter_array = np.array(diameter)
# diameterAll = np.array(diameter)
# diameterAll=diameterAll.T
# rmse_errorAll = np.array(rmse_error)
# rmse_errorAll=rmse_errorAll.T


# rmse_error_array = np.array(rmse_error)
# diameter_array = np.array(diameter)
#savemat("aa_matrix.mat", ToFtrue)
#scipy.io.savemat("ToFtrue1.mat", {'data1': ToFtrue},{'data2': ToFpredict})

# %%  Save data to mat file
# res = np.array(diameter)
#saveMatFile = 'testingResult_'+ modelFile[30:]+".mat"
#savemat(saveMatFile, {"ToFtrue": ToFtrue, "ToFpredict": ToFpredict, "csvFilenames": csvFilenames})


