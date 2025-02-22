# In Spyder, Do following steps
# Run
# Configuration per file...
# Remove all variables before execution [Select Checkbox]

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
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn import metrics
from array import array

import h5py
import mat73
import csv
import re
# release the global state.
tf.keras.backend.clear_session()

tStart=time.time()

# %% Reading csv file
csvFilenames = []
diametermix = []
diameter = []
rmse_error = []

with open('filename_testData_all_517_Files.csv', mode ='r')as csvFile:  
  # reading the CSV file
  csvReader = csv.reader(csvFile)
  for row in csvReader:
      csvFilenames.append(row[0])
      #print(lines)
      
for ii in range(520):
    testFile = csvFilenames[ii]
    diametermix.append(testFile[28:35])
    diameter.append(re.findall("\d+\.\d+", diametermix[ii]))
    print(testFile)
    # %% Testing the model
    modelFile = "ctrainingSmallDia_data_Rec_7to11_Din_25.0%_sortedDia_Din_32.21_to_75.63_files_97_10032023_run5"
    
    #testFile = "test_received_waveforms_Din_33.9_Dout_46.6TEL_2e-6_thickness_6.35mm_06272023.mat.mat";
    outputFile = testFile[0:-4]+".txt";
    
    model = load_model(modelFile)
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
        
        #print('Accuracy: %.2f' % (model.score(x_test[i,:], testLabels[:,setId])*100), '%')
        #precision = metrics.accuracy_score(testLabels, x_test)*100
        #print('sum of testsample',max(testLabels[:,setId]))
        
        if (max(testLabels[:,setId])>0):
            testLabels[:,setId]=testLabels[:,setId]/max(testLabels[:,setId])
            
        # ax = fig.add_subplot(5, 3, setId+1)  
        # ax.plot(np.squeeze(frameTimes)*1e6,testLabels[:,setId]/max(testLabels[:,setId]), 
        #         'b', label='Test Labels Predict')
        # ax.plot(t*1e6,data_test/max(abs(data_test)), 'g', label='Data Test')
        # ax.legend(loc='best'); ax.set_xlabel("Time [$\mu$s]");
        
        # if np.shape(testLabelsTrue)[1]<2:
            # ax.plot(t*1e6,testLabelsTrue/max(testLabelsTrue), 'r',
            #         label='Test Labels True')
            # ax.legend(loc='best');ax.set_xlabel("Time [$\mu$s]");
        # else:
    #         ax.plot(t*1e6,testLabelsTrue[:,setId]/max(testLabelsTrue[:,setId]), 
    #                 'r', label='Test Labels True')
    #         ax.legend(loc='best'); ax.set_xlabel("Time [$\mu$s]");
                
    #     plt.ylim([0,1.5]) 
        
    # plt.tight_layout()
    
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
        #print(idx_tt[i],idx_tltnormax[i])
    
    idx_tt= np.array(idx_tt)
    
    y_predicted= np.array(idx_tt)*1e6;           
    y_actual  = np.array(idx_tltnormax)[:,0,0]*1e6; 
    
    # Plotting
    # plt.figure(figsize=(9, 6))
    # plt.subplot()
    # plt.plot(np.arange(N)+1,y_actual,'*r', 
    #          markersize=12,label='test labels actual')
    # plt.plot(np.arange(N)+1,y_predicted,'ok',
    #          markersize=12,label='test labels predicted')
    # plt.xlabel('Data', fontsize=14, color='black')
    # plt.ylabel("Time [$\mu$s]", fontsize=14, color='black')
    # plt.legend(loc='upper left')
    # plt.show()
    
    # %% Relative error
    relative_error = [0]*len(testLabels[1]) # np.empty(N)
    y_actual  = y_actual.reshape((15, 1))
    relative_error = np.abs(y_actual - y_predicted ) / y_actual
    
    MSE = np.square(np.subtract(y_actual[5:10],y_predicted[5:10])).mean()
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error:\t",'%.3f'%(RMSE))
    
    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(N)+1,relative_error,'ok', 
    #          markersize=12,label='test labels true')
    # plt.xlabel('Receiver', fontsize=14, color='black')
    # plt.ylabel("Absolute Relative Error", fontsize=14, color='black')
    # plt.text(0,0.01,f'RMSE={RMSE:.3f}',fontsize=18,color='xkcd:brick red',
    #          fontweight='bold',fontstyle='italic')
    # plt.show()
    # %% Plot slected receivers only
    
    # plt.figure(figsize=(9, 6))
    # plt.subplot()
    # plt.plot(np.arange(5)+1,y_actual[5:10],'*r', 
    #          markersize=12,label='test labels actual')
    # plt.plot(np.arange(5)+1,y_predicted[5:10],'ok',
    #          markersize=12,label='test labels predicted')
    # plt.xlabel('Data', fontsize=14, color='black')
    # plt.ylabel("Time [$\mu$s]", fontsize=14, color='black')
    # plt.legend(loc='upper left')
    # plt.ylim([0,120]) 
    # plt.show()
    
    # aa=np.array(relative_error[5:10])
    # aa=np.reshape(aa,[5,1])
    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(5)+1, aa,'ok', 
    #          markersize=12,label='test labels true')
    # plt.xlabel('Receiver', fontsize=14, color='black')
    # plt.ylabel("Absolute Relative Error", fontsize=14, color='black')
    # plt.text(1.0,0.01,f'RMSE={RMSE:.3f}',fontsize=18,color='xkcd:brick red',
    #           fontweight='bold',fontstyle='italic')
    # plt.ylim([0,2]) 
    # plt.show()
    
    rmse_error.append(RMSE)
# %% Error metrics
# Mean Square Error(MSE): average squared difference between the predicted and 
# the actual value of a feature or variable.
# Root Mean Square Error(RMSE):RMSE is an acronym for Root Mean Square Error, 
# which is the square root of value obtained from Mean Square Error function.
# Usually, a RMSE score of less than 180 is considered a good score for a 
# moderately or well working algorithm.  In case, the RMSE value exceeds 180, 
# we need to perform feature selection and hyper parameter tuning on the 
# parameters of the model
# R-square;Accuracy

# MSE = np.square(np.subtract(y_actual[7:12],y_predicted[7:12])).mean()
# RMSE = math.sqrt(MSE)
# print("Root Mean Square Error:\t",'%.3f'%(RMSE))
# %% 
# %% 
"""
xdata  = np.array(idx_tt)*1e6
ydata  = np.array(idx_tltnormax)[:,0,0]*1e6
ydata = ydata.reshape((15, 1))

# Compute the Gaussian process fit
from sklearn.gaussian_process import GaussianProcessRegressor
# Compute the Gaussian process fit
gp = GaussianProcessRegressor()
gp.fit(xdata, ydata)

xfit = np.linspace(0, 130, 1000)
yfit, dyfit_ori = gp.predict(xfit[:, np.newaxis],return_std=True)
dyfit = 3 * dyfit_ori  # 2*sigma ~ 95% confidence region

# Visualize the result
plt.plot(xdata, ydata, 'or')
plt.plot(np.arange(130),np.arange(130),'-k')
plt.plot(xfit, yfit, '-', color='gray')
plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,color='gray', alpha=0.2)

# %% Standardized Residuals: A residual is the difference between an observed 
# value and a predicted value in a regression model.
#Residual = Observed value – Predicted value
import pandas as pd
import statsmodels.api as sm
#create dataset
df = pd.DataFrame({'x': xdata[:,1].T,'y': ydata})
#define response variable
y = df['y']

#define explanatory variable
x = df['x']

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit() 
#create instance of influence
influence = model.get_influence()

#obtain standardized residuals
standardized_residuals = influence.resid_studentized_internal

#display standardized residuals
print(standardized_residuals)
plt.scatter(df.x, standardized_residuals,color='black')
plt.xlabel('Time [$\mu$s]')
plt.ylabel('Standardized Residuals')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.show()

"""
# %% export data for further post processing
'''
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
'''

