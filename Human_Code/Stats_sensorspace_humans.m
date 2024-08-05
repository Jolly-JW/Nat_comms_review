%% For stats of human sensor data. Note here we are running the stats on an individual, however the same process 
%applites when programming for more than one subject. 

% Requirment: 
%Combined_sub_example.mat
%chanlocs_prop256.sfp
%Fieldtrip (https://www.fieldtriptoolbox.org/)

%Time: 5 minutes

%Output: Statistical output

%% Clear all
clear all
close all

%Load headmap
ft_defaults
cfg = [];
lay.elec = ft_read_sens('C:\Users\jweh7145\Documents\chanlocs_prop256.sfp');%This will vary by your file location
layout = ft_prepare_layout(cfg, lay); %Setup layout
cfg = [];
cfg.layout = layout;
cfg.method = 'triangulation';
neighbours = ft_prepare_neighbours(cfg); %Setup the neighbours

load('Combined_sub_example.mat');
cfg = []; 
cfg.keeptrials = 'yes';
data = ft_timelockanalysis(cfg, data_combo);

%% First define channels outside those on the head
channels_bad = {'all', '-E241', '-E242', '-E244', '-E243', '-E245', '-E248', ...
        '-E246', '-E247', '-E249', '-E252', '-E250', '-E253', '-E254',...
        '-E251', '-E255', '-E67', '-E73', '-E256', '-E82', '-E91', ...
        '-E238', '-E239', '-E240', '-E234', '-E235', '-E236', '-E237', '-E230', ...
        '-E231', '-E226', '-E232', '-E225', '-E227', '-E233', '-E219', '-E228', '-E218',...
        '-E229', '-E217', '-E216', '-E92', '-E93', '-E102', '-E103', '-E104','-E111', ...
        '-E112', '-E113', '-E120', '-E121', '-E122', '-E133' '-E134', '-E135', '-E145', ...
        '-E146', '-E147', '-E156', '-E165', '-E157', '-E166', '-E174', '-E167', '-E175', ...
        '-E187', '-E176', '-E188', '-E199', '-E189', '-E200', '-E201', '-E208', '-E209'};
    
%Setting up the statistics
cfg = [];
cfg.statistic        = 'ft_statfun_indepsamplesregrT'; %depsamples if using more subjects
cfg.method           = 'montecarlo';
cfg.numrandomization = 10000;
cfg.parameter = 'trial';
cfg.computecritval = 'yes';
cfg.method           = 'montecarlo';
cfg.clusteralpha     = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan        = 2;
cfg.tail             = 0;
cfg.clustertail      = 0;
cfg.alpha            = 0.025;
cfg.correctm='cluster';
cfg.neighbours = neighbours;
cfg.parameter = 'trial';
cfg.clusterthreshold = 'nonparametric_individual';
cfg.design = data.trialinfo(:,1); %PE2 values;

%Select the relevant ariables from above
%cfg.uvar            = 2; %This line is only if you are using multiple subjects (need to define in design)
cfg.ivar            = 1;
%stats_PE2 = ft_timelockstatistics(cfg, data{:,1}); %if more than one dataset held in cell array
stats_PE2 = ft_timelockstatistics(cfg, data);

%% From here, can image the results by taking the mean of the EEG activity in a given region, e.g.,
%To take the mean of the electrodes 12 to 24, and find just the time course of those (you could divide
%up by PE2 values as well for example).: 
 
mean_activity = mean(mean(data, 3)(:,12:24,:),2) 

%These were then exported for plotting. 
