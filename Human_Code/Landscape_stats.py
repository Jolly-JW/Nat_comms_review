#For running cluster stats for MSD (example for wake data)

#Requirements:
#single_subject_matrix_reg folder

#Run time: 1 minute approximately

#Output: Statistical thresholds and images

#%% Import data into python
import numpy as np
import mne
import  scipy.io as spio
import matplotlib.pyplot as plt
import glob
from scipy import stats as stats
from sklearn.feature_extraction.image import grid_to_graph  # noqa: E402
from mne.stats import ttest_1samp_no_p
from functools import partial
from scipy import sparse

#These files aren't too big, so use the folder 'single_subject_matrix_reg'
files_w = glob.glob('E:/Landscape_redo/single_subject_matrix_reg/*_W_*mat')
#Create empty 
wake = np.zeros((21,25,51))
wake_t = np.zeros((21,25,51))
wake_p = np.zeros((21,25,51))
a= 0

#Load and store each
for i in files_w:
    mat1 = spio.loadmat(i)
    wake[a,:,:] = mat1['store_trial']
    wake_t[a,:,:] = mat1['store_trial_t']
    wake_p[a,:,:] = mat1['store_trial_p']
    a = a+1
    
wdc_t = np.swapaxes(wdc_t, 1, 2)
#%% setup clustering

df = 20  # degrees of freedom
t_thresh = stats.distributions.t.ppf(1 - .05 / 2, df=df)

mini_adjacency = grid_to_graph(25, 51).toarray()
assert mini_adjacency.shape == (25*51, 25*51)
adjacency = sparse.csr_matrix(mini_adjacency)  
    
threshold_tfce = dict(start=0, step=0.3)

sigma = 1e-3
stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)

#%% Run clustering
clu_w = mne.stats.permutation_cluster_1samp_test(wake_t, threshold=threshold_tfce, n_permutations=10000, tail=0, stat_fun=None, adjacency=adjacency, max_step=2, exclude=None, step_down_p=0.05, t_power=1, out_type='mask')
print(np.min(clu_w[2]))

#%% Mask the images by significance
wake_p_map = np.zeros((51,25))
wake_t_map = clu_w[0]

for i in range(0, 1275):
    temp1 = clu_w[1][i]
    temp2 = clu_w[2][i]
    wake_p_map[temp1] = temp2
   
wake_mask = np.zeros((51,25))
wake_mask[wake_p_map<.05] = 1
wake_mask[wake_mask==0] = 0.3

#%% Information
print(np.mean(wake_p_map[wake_mask == 1]))
print(np.mean(wake_t_map[wake_mask == 1]))

#%% Plot
wake2= np.mean(wake, 0)

print(np.mean(np.abs(wake2.T[wake_mask == 1])))
print(np.max(np.abs(wake2.T)))

y = [0, 4, 9, 14, 19, 24]
ylabels = [0,10, 20, 30, 40, 50]
x = [0, 10, 20, 30, 40, 50]
xlabels = [0, 10, 20, 30, 40, 50]
fig, axs = plt.subplots(2,3, layout='constrained')

scoreim = axs[0,0].imshow(wake2.T, aspect="auto", cmap='jet', interpolation= 'quadric')
axs[0,0].set_yticks(x, xlabels)
axs[0,0].set_xticks(y, ylabels)
axs[0,0].set_ylabel("Mean Squared Displacement")

cbar = fig.colorbar(scoreim, location='bottom', shrink=0.6, pad = 0.0, ax=axs[1,:])
cbar.ax.set_ylabel('Unit increase in PE2 effect', rotation=0,  labelpad=-325, loc = 'top')

scoreim = axs[1,0].imshow(wake2.T, aspect="auto", cmap='jet', alpha= wake_mask, interpolation= 'spline16')
axs[1,0].set_yticks(x, xlabels)
axs[1,0].set_xticks(y, ylabels)
axs[1,0].set_xlabel("Time in future (ms)")
axs[1,0].set_ylabel("Mean Squared Displacement")
