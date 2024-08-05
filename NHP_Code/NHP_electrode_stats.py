#This file runs the stats on FEF0 in the NHP data.

#Requirment: 
#R_FEF0.mat, W_FEF0.mat

#Time: 3 minutes

#Output: Statistics and images

#%% Import packages
# %matplotlib qt
import mne
import scipy.io
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy 
from scipy.stats import sem

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

#%% Run cluster based stats
len1 = np.size(Temp1, 0)
len2 = np.size(Temp2, 0)
if len1 > len2:
    temp_ind1 = random.sample(range(np.size(Temp1, 0)), len2)
    Temp1 = Temp1[temp_ind1,:]
if len2 > len1:
    temp_ind2 = random.sample(range(np.size(Temp2, 0)), len1)
    Temp2 = Temp2[temp_ind2,:]

n_conditions = 2
#n_observations = np.size(Temp1, 0) + np.size(Temp2, 0)
n_observations = np.size(Temp1, 0)

dfn = n_conditions - 1  # degrees of freedom numerator
dfd = n_observations - n_conditions  # degrees of freedom denominator
thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution

X = [Temp1[:,100:600], Temp2[:,100:600]]
#X = [Temp1, Temp2]

T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
    X,
    n_permutations=n_permutations,
    threshold=thresh,
    tail=0,
    n_jobs=None,
    out_type="mask"
)
#np.sum(T_obs[460:501])
    #%%
fig, axs = plt.subplots(2,1)
fig.suptitle('Wake')

times = np.linspace(-100,400, 500)
for i_c, c in enumerate(clusters):
    c = c[0]
    #if cluster_p_values[i_c] <= 0.001:
    if cluster_p_values[i_c] < 0.05:
        if times[c.start] > 0:
            h = axs[0].axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
   
hf = axs[0].plot(times, T_obs, "g")
axs[0].legend((h,), ("cluster p-value = 0.05",), loc = 'upper left')
axs[0].set_xlabel("time (ms)")
axs[0].set_ylabel("f-values")
axs[0].set_title('Cluster-based stats')
axs[0].plot(np.repeat(0, 50), np.linspace(0,100, 50), 'k--')

axs[1].set_title('Response')
axs[1].plot(np.linspace(-100,400, 500), np.mean(Temp1, 0)[100:600], 'b',label = 'Standard')
axs[1].plot(np.linspace(-100,400, 500).astype(int), np.mean(Temp2, 0)[100:600], 'r',label = 'Oddball')
axs[1].fill_between(np.linspace(-100,400,500), 
                    np.mean(Temp1, 0)[100:600]-sem(Temp1, 0)[100:600], np.mean(Temp1, 0)[100:600]+sem(Temp1, 0)[100:600], alpha = 0.4)
axs[1].fill_between(np.linspace(-100,400,500), 
                    np.mean(Temp2, 0)[100:600]-sem(Temp2, 0)[100:600], np.mean(Temp2, 0)[100:600]+sem(Temp2, 0)[100:600], alpha = 0.4)
axs[1].plot(np.linspace(-100,400, int(500/10)+1).astype(int), np.repeat(0, 51), 'k--')
axs[1].plot(np.repeat(0, 51), np.linspace(-.015,.015, 51), 'k--')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Amplitude')
#axs[1].legend(loc='lower left')
axs[1].set_ylim(-.015, 0.015)

fig.tight_layout()

#%% Plot 2
fig, axs = plt.subplots(1,1)

times = np.linspace(-100,400, 500)
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        if times[c.start] > 0:
            h = axs.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
axs.plot(np.linspace(-100,400, 500), np.mean(Temp1, 0)[100:600], 'b',label = 'Standard')
axs.plot(np.linspace(-100,400, 500).astype(int), np.mean(Temp2, 0)[100:600], 'r',label = 'Oddball')
axs.fill_between(np.linspace(-100,400,500), 
                    np.mean(Temp1, 0)[100:600]-sem(Temp1, 0)[100:600], np.mean(Temp1, 0)[100:600]+sem(Temp1, 0)[100:600], alpha = 0.4)
axs.fill_between(np.linspace(-100,400,500), 
                    np.mean(Temp2, 0)[100:600]-sem(Temp2, 0)[100:600], np.mean(Temp2, 0)[100:600]+sem(Temp2, 0)[100:600], alpha = 0.4)
axs.plot(np.linspace(-100,400, int(500/10)+1).astype(int), np.repeat(0, 51), 'k--')
axs.plot(np.repeat(0, 51), np.linspace(-.015,.015, 51), 'k--')
axs.set_xlabel('Time (ms)')
axs.set_ylabel('Amplitude')
#axs.legend(loc='lower left')
axs.set_ylim(-.015, 0.015)

fig.tight_layout()

