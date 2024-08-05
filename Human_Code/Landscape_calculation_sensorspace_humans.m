%% Setup

% Requirment: 
%Sub1_session1.mat
%chanlocs_prop256.sfp
%Fieldtrip (https://www.fieldtriptoolbox.org/)
%Ket_config.m
%Pull the following: https://github.com/elimuller/Corticothalamic_CM_model

%Time: 10 seconds per subject

%Output: CSV file of MSD x Time x Energy

%Setup
clear all
ft_defaults
percent = 0.25; %Decide on percentiles high and low
decimate_data = 0; %Whether to downsample or not
initialTimepoints = 40; %at times, or could just use samples, 25 = 0, 50 = 100 ms
endtimepoint = 90; %75 = 200 ms
%how many timepoints in the future do you want to calculate
ndt_start = 1;
ndt = 25; %25 = 100 ms at 250hz
%what is the range of MSD you are interested in 
ds = 0:1:50; % the msd range calculated across
normalise = 0; %Whether or not to normalize data
%% Get layout
cfg = [];
lay.elec = ft_read_sens('C:\Users\jweh7145\Documents\Ketamine\chanlocs_prop256.sfp');
layout = ft_prepare_layout(cfg, lay);
%Get neighbours
cfg = [];
cfg.layout = layout;
cfg.method = 'triangulation';
neighbours = ft_prepare_neighbours(cfg);

layout.pos(length(layout.pos)-1:end, :) = [];
nb_mat = squareform(pdist(layout.pos));
cutoff = 0.1;
electrode_neighbours =  (nb_mat < cutoff);

channels_bad = {'all', '-E241', '-E242', '-E244', '-E243', '-E245', '-E248', ...
        '-E246', '-E247', '-E249', '-E252', '-E250', '-E253', '-E254',...
        '-E251', '-E255', '-E67', '-E73', '-E256', '-E82', '-E91', ...
        '-E238', '-E239', '-E240', '-E234', '-E235', '-E236', '-E237', '-E230', ...
        '-E231', '-E226', '-E232', '-E225', '-E227', '-E233', '-E219', '-E228', '-E218',...
        '-E229', '-E217', '-E216', '-E92', '-E93', '-E102', '-E103', '-E104','-E111', ...
        '-E112', '-E113', '-E120', '-E121', '-E122', '-E133' '-E134', '-E135', '-E145', ...
        '-E146', '-E147', '-E156', '-E165', '-E157', '-E166', '-E174', '-E167', '-E175', ...
        '-E187', '-E176', '-E188', '-E199', '-E189', '-E200', '-E201', '-E208', '-E209'};

%% Run through
subject_keeper = {};

%We will runthrough and combine this across subjects in difference
%conditions, but as an example: 
load('Combined_sub_example.mat')
sub_num = 1;
%Setting some stuff up and initializing variables
z=1;
x=1;
trials = []; 
trials_info = [];
trials2 = [];
trials_info2 = []; 
       
data_wake = data_combo;
index_WH = find([data_wake.trialinfo(:, 2)].' >  quantile([data_wake.trialinfo(:, 2)].', 0.75));
index_WL = find([data_wake.trialinfo(:, 2)].' <  quantile([data_wake.trialinfo(:, 2)].', 0.25));

%Extracting the highest and lowest 25% of PE2 trials
cfg = [];
cfg.keeptrials = 'yes';
cfg.channel = channels_bad;
cfg.trials = index_WH;
data_WH = ft_timelockanalysis(cfg, data_wake);
cfg.trials = index_WL;
data_WL = ft_timelockanalysis(cfg, data_wake);
subject_keeper{1, sub_num} = data_WH.trial; 
subject_keeper{2, sub_num} = data_WL.trial; 
Combo = {};
for i = 1:length(WH)
    temp = WH(:,i);
    Combo{i,1} = cell2mat(temp);
    temp = WL(:,i);
    Combo{i,2} = cell2mat(temp);
end

%% From here running the landscape analysis 
nrgMsdDt_WH = []; %First on WH, then WL etc, only WH calculation shown, but same for all variables
a = 1;
for i = 1:size(Combo,1)
    data_temp = Combo{i,1};
    nrgMsdDt_combo = [];
    for q = 1:size(data_temp,1)
        Sig = squeeze(data_temp(q,:,:))';
        nrgMsdDt = [];
        for dt = ndt_start:ndt

            %Mean-Squared Displacement calculation
            %The MSD is how much your signal has changed at a given time-delayinitial timepoint T, it is
            %between T and T+dt

            %lets first calculate the displacement between timepoints interested
            D = Sig(initialTimepoints+dt,:)-Sig(initialTimepoints,:);
            %now square it
            SD = D.^2;
            %now take the mean
            MSD = mean(SD,2);

            %MSD should be a vector double check
            [nrg] = PdistGaussKern(MSD,ds);

            %fill up the empty variable
            nrgMsdDt(dt,:) = nrg;
        end
        nrgMsdDt_WH(a,:,:) = nrgMsdDt;
        a = a +1;
    end 
end

%% then save
save('E:\DC_three_drug\energy_store.mat', 'nrgMsdDt_WH');