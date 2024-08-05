#For running correlations between Matrix thalamus connection and each voxel

#Requirements: 
#stc_temp.pkl
#PV_Ca_Corr.csv
#Wake_data.mat
#rh.Schaefer2018_400Parcels_Kong2022_17Networks_order.annot
#lh.Schaefer2018_400Parcels_Kong2022_17Networks_order.annot
#Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv

#Run time: 3 minutes

#Output: statistical output

#%% Import packages
import scipy.io as spio
from mne.datasets import sample
import os.path as op
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

#Setup path
data_path = sample.data_path()
subjects_dir = data_path / "subjects"
src_fname = subjects_dir / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif"
subject = "sample"

#Load STC
file_name ='C:/Users/jweh7145/Documents/Source_Recons/stc_temp.pkl'
with open(file_name, 'rb') as f:
    stc = pkl.load(f)
f.close

#Load the wake and DC data (held as mats)
Wake = spio.loadmat('E:/DC_three_drug/Wake_data.mat')

#Separate data
W_High = Wake['X_high']
W_Low = Wake['X_low']

#Find the peak latency
W_High_ind = np.argmax(np.abs(W_High[:,50:75,:]), axis = 1) #50-75 = samples of interest
W_Low_ind = np.argmax(np.abs(W_Low[:,50:75,:]), axis = 1) 
#And  amplitudes
W_High_ind2 = np.max(np.abs(W_High[:,50:75,:]), axis = 1) 
W_Low_ind2= np.max(np.abs(W_Low[:,50:75,:]), axis = 1) 

a = W_High_ind[0,:]
a2 = np.repeat(a[:, np.newaxis], 124, axis=1)

stc.data= a2

#%% Surface
import mne
from mne.datasets import fetch_fsaverage
fs_dir = fetch_fsaverage(verbose=True)
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')

subjects_dir =  'E:/DC_three_drug/Matrix Thalamus/Parcellations/FreeSurfer5.3/'
labels = mne.read_labels_from_annot('fsaverage', parc='aparc', hemi='both', surf_name='pial',subjects_dir=subjects_dir, verbose=True,
    annot_fname = 'E:/DC_three_drug/Matrix Thalamus/Parcellations/FreeSurfer5.3/fsaverage/label/rh.Schaefer2018_400Parcels_Kong2022_17Networks_order.annot')

Brain = mne.viz.get_brain_class()
brain = Brain(
    "fsaverage",
    "rh",
    "inflated",
    subjects_dir=subjects_dir,
    cortex="low_contrast",
    background="white",
    size=(800, 600),
)
brain.add_annotation('E:/DC_three_drug/Matrix Thalamus/Parcellations/FreeSurfer5.3/fsaverage/label/rh.Schaefer2018_400Parcels_Kong2022_17Networks_order.annot')
brain = Brain(
    "fsaverage",
    "lh",
    "inflated",
    subjects_dir=subjects_dir,
    cortex="low_contrast",
    background="white",
    size=(800, 600),
)
brain.add_annotation('E:/DC_three_drug/Matrix Thalamus/Parcellations/FreeSurfer5.3/fsaverage/label/lh.Schaefer2018_400Parcels_Kong2022_17Networks_order.annot')

#%%
#Finding MNI coords of our space
subjects_dir2 = op.dirname(fs_dir)
mni_coords_lh = mne.vertex_to_mni(np.int0(np.linspace(0,10241,10242)), hemis= 0, subject = 'fsaverage', subjects_dir=subjects_dir2, verbose=None)
mni_coords_rh = mne.vertex_to_mni(np.int0(np.linspace(0,10241,10242)), hemis= 1, subject = 'fsaverage', subjects_dir=subjects_dir2, verbose=None)
mni_coord_combine = np.concatenate((mni_coords_lh, mni_coords_rh))
#find a centroid by voxel distance measure
#RAS is only for the other version
#centroids = pd.read_csv('E:/DC_three_drug/Matrix Thalamus/Parcellations/MNI/Centroid_coordinates/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
centroids = pd.read_csv('E:/DC_three_drug/Matrix Thalamus/Schaefer400_xyz.csv')
distances =np.zeros((len(centroids), 20484))
for i in range(0, len(centroids)): #looping throuhg LH
    #centroid = (centroids['R'][i], centroids['A'][i], centroids['S'][i])
    centroid = (centroids['X'][i], centroids['Y'][i], centroids['Z'][i])
    for x in range(0, len(mni_coord_combine)):
        dif = centroid - mni_coord_combine[x,:]
        distance = np.sqrt(np.sum(np.square(dif)))
        distances[i,x] = distance
        
#Now for each label, we will find which voxels we want to take (so it'll be a 400 long space)
indices = list()
for i in range(0, len(centroids)):
    temp = distances[i,:]
    temp2 = np.argwhere(temp<10)
    indices.append(temp2)
    
#Find the averages of each space
results_WH =np.zeros((len(centroids), 21))
results_WL = np.zeros((len(centroids), 21))

results_WH2 =np.zeros((len(centroids), 21))
results_WL2 = np.zeros((len(centroids), 21))

for i in range(0, len(centroids)):
    index_vals = indices[i]
    temp_W_H = np.mean(np.squeeze(W_High_ind[:,index_vals]),1)
    temp_W_L = np.mean(np.squeeze(W_Low_ind[:,index_vals]),1)
    
        results_WH[i,:] = temp_W_H
    results_WL[i,:] = temp_W_L
    
    #And for amplitudes
    temp_W_H2 = np.mean(np.squeeze(W_High_ind2[:,index_vals]),1)
    temp_W_L2 = np.mean(np.squeeze(W_Low_ind2[:,index_vals]),1)
    
    results_WH2[i,:] = temp_W_H2
    results_WL2[i,:] = temp_W_L2
  
#Save the data
np.savetxt("E:/DC_three_drug/Matrix Thalamus/WH_lat_schaf2.csv", results_WH, delimiter=",")
np.savetxt("E:/DC_three_drug/Matrix Thalamus/WL_lat_schaf2.csv", results_WL, delimiter=",")

np.savetxt("E:/DC_three_drug/Matrix Thalamus/WH_lat_schaf2_amp.csv", results_WH2, delimiter=",")
np.savetxt("E:/DC_three_drug/Matrix Thalamus/WL_lat_schaf2_amp.csv", results_WL2, delimiter=",")

#%% Now to find which areas we want to keep
values_matrix = pd.read_csv('E:/DC_three_drug/Matrix Thalamus/PV_Ca_Corr.csv')

#%%
#Plottin the means
results_WH_M = np.mean(results_WH, 1)
results_WL_M = np.mean(results_WL, 1)
values = np.array(values_matrix['Diff_percent'])

#Based on threshold
results_WH_M = np.mean(results_WH[values_matrix['Diff_percent']>0.4], 1)
results_WL_M = np.mean(results_WL[values_matrix['Diff_percent']>0.4], 1)
values = np.array(values_matrix['Diff_percent'][values_matrix['Diff_percent']>0.4])

#Based on threshold
values_sd = np.std(np.array(values_matrix['Diff_percent']))
values_mean = np.mean(np.array(values_matrix['Diff_percent']))
thres1 = values_mean+2*values_sd
thres2 = values_mean-2*values_sd

results_WH_M = np.mean(results_WH[(values_matrix['Diff_percent']>thres2) & (values_matrix['Diff_percent']<thres1)], 1)
results_WL_M = np.mean(results_WL[(values_matrix['Diff_percent']>thres2) & (values_matrix['Diff_percent']<thres1)], 1)
values = np.array(values_matrix['Diff_percent'][(values_matrix['Diff_percent']>thres2) & (values_matrix['Diff_percent']<thres1)])

#Based on ROI
rows = [194,195,196,197,198,199,390,391,392,393,394,395,396,397,398,399] #Audio plots
rows = [194,195,196,197,198,199,390,391,392,393,394,395,396,397,398,399] #Audio plots + default b

results_temp = results_WH[rows,:]
results_temp = results_WL[rows,:]
results_WH_M = np.mean(results_temp, 1)
results_WL_M = np.mean(results_temp, 1)
values = np.array(values_matrix['Diff_percent'][rows])

#Plot from here
m_WH, b_WH = np.polyfit(values, results_WH_M, 1)
m_WL, b_WL = np.polyfit(values, results_WL_M, 1)

fig, axs = plt.subplots(2,2)
fig.suptitle('Vertically stacked subplots')
axs[0,0].scatter(values,results_WH_M)
axs[0,0].plot(values, m_WH*values+b_WH)

axs[0,1].scatter(values,results_WL_M)
axs[0,1].plot(values, m_WL*values+b_WL)

axs[0,0].set_xlabel('Percent Matrix')
axs[0,0].set_ylabel('Mean Peak Latency')
axs[0,0].set_title('Wake High PE2')
axs[0,1].set_title('Wake Low PE2')
axs[0,0].set_ylim([16, 26])
axs[0,1].set_ylim([16, 26])
plt.tight_layout()


