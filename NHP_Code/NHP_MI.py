#This file runs the MI calculations for the NHPs, then plot it and run decoding

#Requirment: 
#M1_FEF0.mat, M2_FEF0.mat
#Note, to cut down on files and size, we are 'imagining' the connection between FEF0 standard and FEF0 wake
#While the actual MI calcuations were based on FEF0 to FEF1, for example.

#Time: 3 minutes

#Output: Statistics and images

#%% load packages
#%matplotlib qt
import scipy.io
import glob
import numpy as np
import random
import scipy 
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
time_start = (np.linspace(100,600, int(500/10)+1)).astype(int)

#%% Load all data (if I want to go this way instead)
files = glob.glob("C:/Users/jorda/Documents/For_sharing/*FEF*.mat")

mat = scipy.io.loadmat(files[0]) #SELECT THESE NUMBERS TO SELECT FILES (0,5) (3,4) (1,7), (2,6)
mat2 = scipy.io.loadmat(files[1]) #SELECT THESE NUMBERS TO SELECT FILES
Temp1 = np.concatenate((mat['SW'], mat2['SW']),0) 
Temp2 = np.concatenate((mat['OW'], mat2['OW']),0) 

pval = 0.025  # arbitrary .01
n_permutations = 20000 #20000
std_lim = 2.5 #2 is normal 
random.seed(10)

#%% Try cleaning Temp1
#SD
std_check = np.std(Temp1, 1)
std_check_sd = np.std(std_check)
std_check_mean = np.mean(std_check)
std_idx1 = np.where(std_check < std_check_mean + std_check_sd*std_lim)[0]
std_idx2 = np.where(std_check > std_check_mean - std_check_sd*std_lim)[0]
std_idx = np.intersect1d(std_idx1, std_idx2)
Temp1 = Temp1[std_idx, :] #Clean out

#Max min
abs_check = np.max(np.abs(Temp1), 1)
abs_std = np.std(abs_check)
abs_mean = np.mean(abs_check)
abs_idx = np.where(abs_check< abs_mean+abs_std*std_lim)[0]
Temp1 = Temp1[abs_idx, :] #Clean out

#Baseline jump
baseline = np.mean(Temp1[:, 100:200], 1) 
post_baseline = np.max(Temp1[:, 201:250], 1)
diff =np.abs((post_baseline-baseline))
mean_diff = np.mean(diff)
sd_diff = np.std(diff)
diff_idx = np.where(diff < (mean_diff)*std_lim)[0]
Temp1 = Temp1[diff_idx,:]  #Clean out

#Alpha power
fft_vals = np.absolute(np.fft.rfft(Temp1[:,200::], axis = 1)) #200 for from baseline
fft_freq = np.fft.rfftfreq(np.size(Temp1[:,200::],1), 1.0/1000)
fft_vals =fft_vals[:,np.where((fft_freq>8) & (fft_freq<12))[0]]
mean_pow = np.mean(fft_vals,1)
mean_pow_std = np.std(mean_pow)
mean_pow_mean = np.mean(mean_pow)
pow_idx = np.where(mean_pow < mean_pow_mean+ std_lim*mean_pow_std)[0]
Temp1 = Temp1[pow_idx,:]  #Clean out

#Temp 2 data ====================
#SD
std_check = np.std(Temp2, 1)
std_check_sd = np.std(std_check)
std_check_mean = np.mean(std_check)
std_idx1 = np.where(std_check < std_check_mean + std_check_sd*std_lim)[0]
std_idx2 = np.where(std_check > std_check_mean - std_check_sd*std_lim)[0]
std_idx = np.intersect1d(std_idx1, std_idx2)
Temp2 = Temp2[std_idx, :] #Clean out

#Max min
abs_check = np.max(np.abs(Temp2), 1)
abs_std = np.std(abs_check)
abs_mean = np.mean(abs_check)
abs_idx = np.where(abs_check< abs_mean+abs_std*std_lim)[0]
Temp2 = Temp2[abs_idx, :] #Clean out

#Baseline jump
baseline = np.mean(Temp2[:, 100:200], 1) 
post_baseline = np.max(Temp2[:, 201:250], 1)
diff =np.abs(baseline-post_baseline)
mean_diff = np.mean(diff)
sd_diff = np.std(diff)
diff_idx = np.where(diff < (mean_diff)*std_lim)[0]
Temp2 = Temp2[diff_idx,:]  #Clean out

#Alpha power
fft_vals = np.absolute(np.fft.rfft(Temp2[:,200::], axis = 1))
fft_freq = np.fft.rfftfreq(np.size(Temp2[:,200::],1), 1.0/1000)
fft_vals =fft_vals[:,np.where((fft_freq>8) & (fft_freq<12))[0]]
mean_pow = np.mean(fft_vals,1)
mean_pow_std = np.std(mean_pow)
mean_pow_mean = np.mean(mean_pow)
pow_idx = np.where(mean_pow < mean_pow_mean+ std_lim*mean_pow_std)[0]
Temp2 = Temp2[pow_idx,:]  #Clean out

#%% Run MI calc (number of iterations cut down to decrease time for running)
iteration_num = 100
sample_size = 100
future_time = 100
future_time2 = (np.linspace(0,100, int(100/10)+1)).astype(int)

len1 = np.size(Temp2, 0)
len2 = np.size(Temp1, 0)
if len1 > len2:
    temp_ind1 = random.sample(range(np.size(Temp2, 0)), len2)
    Temp2 = Temp2[temp_ind1,:]
if len2 > len1:
    temp_ind2 = random.sample(range(np.size(Temp1, 0)), len1)
    Temp1 = Temp1[temp_ind2,:]

#Initialise variables
score = np.zeros((iteration_num, len(time_start), len(future_time2)))
score_reverse = np.zeros((iteration_num, len(time_start), len(future_time2)))

b = 0 
for x in time_start:
    time_end = x+future_time
    a = 0
    for y in future_time2:
        for i in range(0,iteration_num):
            temp_ind1 = random.sample(range(np.size(Temp1, 0)), sample_size)
            temp_ind2 = random.sample(range(np.size(Temp2, 0)), sample_size)
            
            # One direction
            temp1 = Temp1[temp_ind1,x:time_end]
            temp2 = Temp2[temp_ind2,x+y:time_end+y]
            
            temp1 = np.expand_dims(np.mean(temp1, 0), 1)
            temp2 = np.mean(temp2, 0)
            
            score[i,b,a] = mutual_info_regression(temp1, temp2)
            
            #Other direction
            temp1 = Temp1[temp_ind1,x+y:time_end+y]
            temp2 = Temp2[temp_ind2,x:time_end]
            
            temp2 = np.expand_dims(np.mean(temp2, 0), 1)
            temp1 = np.mean(temp1, 0)
            
            score_reverse[i,b,a] = mutual_info_regression(temp2, temp1)
        
        a = a+1
    b =b+1
print(str(x))
score2 = np.mean(score, 0)

#%% Now plot
FEF1_FEF0_SW = np.mean(score, 0)
FEF1_FEFN1_SW = np.mean(score_reverse, 0)

fig, axs = plt.subplots(2, layout='constrained')

y = [0, 10, 20, 30, 40, 50]
ylabels = [-100, 0, 100, 200, 300, 400]
x = [0, 2, 4, 6, 8, 10]
xlabels = [0, 20, 40, 60, 80, 100]

axs[0].set_title('TimeSeries1 --> TimeSeries2')
scoreim = axs[0].imshow(FEF1_FEF0_SW.T, aspect="auto", vmin=0.59, vmax=1.148, cmap = 'jet', interpolation= 'quadric')
axs[1].set_title('TimeSeries2 --> TimeSeries1')
scoreim = axs[1].imshow(FEF1_FEFN1_SW.T, aspect="auto", vmin=0.59, vmax=1.148, cmap = 'jet', interpolation= 'quadric')

axs[0].set_yticks(x, xlabels)
axs[0].set_xticks(y, ylabels)
axs[0].set_xlabel("Time start (ms)")
axs[0].set_ylabel("Time lag (ms)")
axs[1].set_yticks(x, xlabels)
axs[1].set_xticks(y, ylabels)
axs[1].set_xlabel("Time start (ms)")

#%% And then compare using decoding
#First reshape score
X1 = np.reshape(score, [np.size(score, 0), -1]).copy()
X2 = np.reshape(score_reverse, [np.size(score_reverse, 0), -1]).copy()

score_decode = np.zeros((iteration_num))

for i in range(0,iteration_num):
    temp_ind1 = random.sample(range(np.size(X1, 0)), sample_size)
    temp_ind2 = random.sample(range(np.size(X2, 0)), sample_size)
   
    temp1 = X1[temp_ind1, :]
    temp2 = X2[temp_ind2,:]
    
    
    X_temp = np.concatenate((temp1, temp2), 0)
    y_temp = np.concatenate((np.ones(sample_size), np.ones(sample_size)*2))
    X_temp, y_temp = shuffle(X_temp, y_temp, random_state = 10)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=10)
    
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=5, random_state=10
    ),)
    clf.fit(X_train, y_train)
    score_decode[i] = clf.score(X_test, y_test)
    
    print(str(i))
    
#Calculate mean and sd
print(np.mean(score_decode))
print(np.std(score_decode))
