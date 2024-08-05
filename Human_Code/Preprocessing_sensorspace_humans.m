
%% For preprocessing of human data

% Requirment: 
%Sub1_session1.mat
%chanlocs_prop256.sfp
%TAPAS (https://github.com/translationalneuromodeling/tapas/tree/master/HGF)
%Fieldtrip (https://www.fieldtriptoolbox.org/)
%Ket_config.m

%Time: 20 seconds per subject

%Output: Mat file for statistics, and can be transferred into MNE format

clear all
close all
ft_defaults

%Setup layout
cfg = [];
lay.elec = ft_read_sens('C:\Users\61415\Documents\Ketamine data\chanlocs_prop256.sfp');
layout = ft_prepare_layout(cfg, lay);

subject_list = dir('E:\Jordan\*.mat'); %This will vary by your folder

for subjects = 1:length(subject_list)
    %% Load 
    name = subject_list(subjects).name;
    load([subject_list(subjects).folder,'\', name]); 
    FTEEG = ftdat;
    FTEEG.trialinfo = FTEEG.event;
    %Proprocess
    cfg = [];
    cfg.reref       = 'yes';
    cfg.channel     = 'all';
    cfg.refchannel  = 'all';
    cfg.lpfilter = 'yes';
    cfg.hpfilter = 'yes';
    cfg.lpfreq = 30;
    cfg.hpfreq = 0.5;
    cfg.lpfilttype = 'but';
    cfg.hpfilttype = 'but';

    %Filter
    data_eeg        = ft_preprocessing(cfg, FTEEG);

    %Estimating the HGF - must have the tapas package
    tones = [data_eeg.trialinfo.tone].';
    est = tapas_fitModel([], tones, 'tapas_hgf_transition_config', 'tapas_bayes_optimal_transition_config');
   
    for i = 2:length(est.u_orig)
           a = est.u_orig(i-1);
           b = est.u_orig(i);
           data_eeg.trialinfo(i).PE2 = (est.traj.epsi(i-1,2,b,a));
           data_eeg.trialinfo(i).PE3 = sum(est.traj.epsi(i-1,3,a,:),'omitnan');       
    end

    %Epoch and baseline correct
    cfg = [];
    cfg.trialdef.prestim    = 0.1; % in seconds
    cfg.trialdef.poststim   = 0.4; % in seconds

    %Because of the way the data was sent to us, we had to run the epoching
    %manually:
    trl = [];
    for i=1:length(data_eeg.trialinfo)
     begsample     = data_eeg.trialinfo(i).latency - cfg.trialdef.prestim*FTEEG.fsample;
     endsample     = data_eeg.trialinfo(i).latency + cfg.trialdef.poststim*FTEEG.fsample-2;
     trl(i, :) = [round([begsample endsample]), (data_eeg.trialinfo(i).tone)];
    end

    data = [];
    data.fsample = FTEEG.fsample;
    data.label = data_eeg.label';
    data.trialinfo = data_eeg.trialinfo;
    data.trialinfo(1) = []; %Deleting first trial
     for i=2:length(data_eeg.trialinfo)
         if (data_eeg.trialinfo(i).latency+100)<length(FTEEG.time{1,1})
             data.trial{1,i-1} = data_eeg.trial{1,1}(:,(trl(i,1)):(trl(i,2)));
             data.time{1,i-1} = data_eeg.time{1,1}(127:250)-0.6;  
             data.sampleinfo(i-1,1) = data_eeg.trialinfo(i).latency-cfg.trialdef.prestim*FTEEG.fsample;
             data.sampleinfo(i-1,2) = data_eeg.trialinfo(i).latency+ cfg.trialdef.poststim*FTEEG.fsample-2;
            % data.timeinfo(i-1,1) = data_eeg.time{1,1}(trl(i,1));
            % data.timeinfo(i-1,2) = data_eeg.time{1,1}(trl(i,2));
            % data.latency(i-1) = FTEEG.event(i).latency;
         end
     end
    
     %Detrend, demean, baseline correct
    cfg = [];
    cfg.detrend = 'yes';
    cfg.demean = 'yes';
    cfg.baselinewindow = [-.1, 0];
    
    data2 = ft_preprocessing(cfg, data);
    data2.trialinfo = data.trialinfo;
    data = data2;


    %% Now run the cleaning

    %Create EOGH and EOGV 
    cfg              = [];
    cfg.channel      = {'E31' 'E238'};
    cfg.reref        = 'yes';
    cfg.implicitref  = [];
    cfg.refchannel   = {'E31'};
    eogh             = ft_preprocessing(cfg, data);

    cfg              = [];
    cfg.channel      = {'E238'};
    eogh             = ft_selectdata(cfg, eogh);
    eogh.label       = {'eogh'};

    % EOGV channel
    cfg              = [];
    cfg.channel      = {'E241' 'E37'};
    cfg.reref        = 'yes';
    cfg.implicitref  = [];
    cfg.refchannel   = {'E37'};
    eogv             = ft_preprocessing(cfg, data);

    cfg              = [];
    cfg.channel      = 'E241';
    eogv             = ft_selectdata(cfg, eogv);
    eogv.label       = {'eogv'};

   
    % only keep all non-EOG channels
    cfg         = [];
    cfg.channel = setdiff(1:256, [31, 238, 241]);        % you can use either strings or numbers as selection
    data2        = ft_selectdata(cfg, data);

    % append the EOGH and EOGV channel to the selected EEG channels
    %cfg  = [];
   % data2 = ft_appenddata(cfg, data2, eogv, eogh);

    %% EOG-based delete
    cfg = [];
    cfg.continuous = 'no';

     % channel selection, cutoff and padding
     cfg.artfctdef.zvalue.channel     = 'eogv';
     cfg.artfctdef.zvalue.cutoff      = 4;
     cfg.artfctdef.zvalue.trlpadding  = 0;
     cfg.artfctdef.zvalue.artpadding  = 0.1;
     cfg.artfctdef.zvalue.fltpadding  = 0.05;

     % algorithmic parameters
     cfg.artfctdef.zvalue.bpfilter   = 'yes';
     cfg.artfctdef.zvalue.bpfilttype = 'but';
     cfg.artfctdef.zvalue.bpfreq     = [2 15];
     cfg.artfctdef.zvalue.bpfiltord  = 4;
     cfg.artfctdef.zvalue.hilbert    = 'yes';

     [cfg, artifact_EOGV] = ft_artifact_zvalue(cfg, eogv);

     cfg = [];
     cfg.continuous = 'no';

     % channel selection, cutoff and padding
     cfg.artfctdef.zvalue.channel     = 'eogh';
     cfg.artfctdef.zvalue.cutoff      = 4;
     cfg.artfctdef.zvalue.trlpadding  = 0;
     cfg.artfctdef.zvalue.artpadding  = 0.1;
     cfg.artfctdef.zvalue.fltpadding  = 0.05;

     % algorithmic parameters
     cfg.artfctdef.zvalue.bpfilter   = 'yes';
     cfg.artfctdef.zvalue.bpfilttype = 'but';
     cfg.artfctdef.zvalue.bpfreq     = [2 15];
     cfg.artfctdef.zvalue.bpfiltord  = 4;
     cfg.artfctdef.zvalue.hilbert    = 'yes';

    %cfg.artfctdef.zvalue.interactive = 'yes';
    [cfg, artifact_EOGH] = ft_artifact_zvalue(cfg, eogh);

    artifact_EOG = [artifact_EOGH; artifact_EOGV];
    %% Jump artefacts
    cfg = [];
    cfg.continuous = 'no';

    % channel selection, cutoff and padding
    cfg.artfctdef.zvalue.channel = 'EEG';
    cfg.artfctdef.zvalue.cutoff = 20;
    cfg.artfctdef.zvalue.trlpadding  = 0;
     cfg.artfctdef.zvalue.artpadding  = 0.1;
     cfg.artfctdef.zvalue.fltpadding  = 0.05;

    % algorithmic parameters
    cfg.artfctdef.zvalue.cumulative = 'yes';
    cfg.artfctdef.zvalue.medianfilter = 'yes';
    cfg.artfctdef.zvalue.medianfiltord = 9;
    cfg.artfctdef.zvalue.absdiff = 'yes';

    % make the process interactive
    %cfg.artfctdef.zvalue.interactive = 'yes';
    [cfg, artifact_jump] = ft_artifact_zvalue(cfg, data2);

    %% Detect trials in which the threshold exceeds some amount
    %doing it manually
    Channel_keeper = [];
    for i = 1:length(data.trial)
       for x = 1:length(data.label)
           z = abs(data.trial{1,i}(x,:));
           if isempty(find(z>80))
               Channel_keeper(x,i) = 0;
           else
               Channel_keeper(x,i) = 1;
           end
       end
    end

    Channel_checker = sum(Channel_keeper, 2);
    Replace = [];
    for i = 1:length(Channel_checker)
       if Channel_checker(i) > 0.2*length(data.trial)
          Replace(i,1) = 1;
       else
           Replace(i,1) = 0;
       end
    end
    
    x = 1;
    y = 1;
    Channels = cell(1);
    Replace_channels=cell(1);
    for i = 1:length(data.label)
       if Replace(i) == 0
           Channels{1,x} = layout.label{i};
           x = x+1;
       else
           Replace_channels{1,y} = layout.label{i};
           y = y+1;
       end
    end
    
    %Delete those channels
    data.channel = data.label;
    cfg = [];
    cfg.channel = Channels;
    data2 = ft_selectdata(cfg, data);
    
    % Now use fieldtrip to get rid of bad trials without the bad channel
     cfg = [];
    cfg.artfctdef.threshold.channel   = 'EEG';
    cfg.artfctdef.threshold.bpfilter  = 'no';
    cfg.artfctdef.threshold.max = 80;
    cfg.artfctdef.threshold.min = -80;
    cfg.artfctdef.threshold.trlpadding  = 0.1;
     cfg.artfctdef.threshold.artpadding  = 0.1;
     cfg.artfctdef.threshold.fltpadding  = 0.05;
    cfg.artfctdef.threshold.interactive = 'yes';
    [cfg, artifact] = ft_artifact_threshold(cfg, data2);
   
    %% Delete data
    start_num = length(trialinfo);
    cfg=[];
    cfg.artfctdef.reject = 'complete'; % this rejects complete trials, use 'partial' if you want to do partial artifact rejection
    cfg.artfctdef.eog.artifact = artifact_EOG; %
    cfg.artfctdef.jump.artifact = artifact_jump;
    cfg.artfctdef.xxx.artifact = artifact;
    data_no_artifacts = ft_rejectartifact(cfg,data);
    
    data = data_no_artifacts;
    
    %Keep track of number of trials deleted, and save data
    del_num = length(artifact)+length(artifact_EOG)+ length(artifact_jump);
    save(name, 'data', 'Replace_channels', 'start_num', 'del_num');
    
end