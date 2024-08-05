%For preprocessing of NHP datasets

%Requirment: 
%NHP_preprocessing_file.mat

%Time: 5 seconds

%Output: data, which can then be turned into fif if desired for MNE, or in
%this case, we just get the data out.

clear all 
close all
load('C:\Users\jorda\Documents\For_sharing\NHP_preprocessing_file.mat')

cfg = [];
cfg.trials = [2:length(ft_temp.trialinfo)];
cfg.demean          = 'yes';
cfg.baselinewindow  = [-0.1, .0];
cfg.detrend = 'no';
cfg.resamplefs = 500;
cfg.continuous = 'no';
data = ft_resampledata(cfg, ft_temp);

cfg= [];
cfg.keeptrials = 'yes';
cfg.latency = [-.1, 400]; %To keep even with human data
data = ft_timelockanalysis(cfg, data);

%This can be saved for import into python, divided by:
labels = data.label;
trialinfo = data.trialinfo;
trials = data.trial;

save("C:\Users\jorda\Documents\For_sharing\file.mat", "trials", 'labels', 'trialinfo');
