% Clean up residuals
clear all;

% -----------------------------------------------------------------------------------------
% This script performs the preprocessing of the EEG data. The result are cleaned data
% of stimulus-locked epochs and R-peak-locked epochs. For the latter, only R-peaks are
% consideres that precede nor follow an experimental (visual) stimulation. The stimulus and
% R-peak-locked data has time x epoch dimensionality.
% -----------------------------------------------------------------------------------------

% Path vars
PATH_RAW             = 'add_path_here';
PATH_ICSET           = 'add_path_here';
PATH_AUTOCLEANED     = 'add_path_here';
PATH_ECG             = 'add_path_here';
PATH_REGRESSION_DATA = 'add_path_here';
PATH_META            = 'add_path_here';
PATH_EEGLAB          = 'add_path_here';
PATH_FORCE           = 'add_path_here';

% Subject list
subject_list = {'VP02', 'VP03', 'VP04', 'VP05', 'VP06', 'VP07', 'VP08', 'VP09', 'VP10', 'VP11',...
                'VP12', 'VP13', 'VP14', 'VP15', 'VP16', 'VP17', 'VP18', 'VP19', 'VP20', 'VP21',...
                'VP22', 'VP23', 'VP24', 'VP25', 'VP26', 'VP27', 'VP28', 'VP29', 'VP30', 'VP31',...
                'VP32', 'VP33', 'VP34', 'VP35', 'VP36', 'VP37', 'VP38', 'VP39', 'VP40', 'VP41'};

% Switch parts of script on/off
to_execute = {'part1'};

% ======================== PART1: Prepare regression data =========================
if ismember('part1', to_execute)

    % Init eeglab and find chanlocfile
    addpath(PATH_EEGLAB);
    eeglab;
    channel_location_file = which('dipfitdefs.m');
    channel_location_file = channel_location_file(1 : end - length('dipfitdefs.m'));
    channel_location_file = [channel_location_file, 'standard_BESA/standard-10-5-cap385.elp'];

    % Iterating subject list
    for s = 1 : length(subject_list)

        % Dataset name
        subject = subject_list{s}; 
        id = str2num(subject(3 : 4));  

        % Load raw data
        EEG = pop_loadbv(PATH_RAW, [subject '.vhdr'], [], []);

        % Get R peak events where the R-peak havent been used
        % for stimulus locking
        EEG.event_R_nolock = get_R_nolock_events(EEG);

        % Get stimulus locked event structure
        EEG.event_stimlocked = get_stim_events(EEG, id, PATH_FORCE, [61, 62]);

        % Drop forcechannels
        EEG = pop_select(EEG, 'nochannel', [61, 62]);

        % Filter the data (EEG and ECG)
        EEG  = pop_basicfilter(EEG, [1 : EEG.nbchan], 'Cutoff', [0.5, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary');

        % Separate ECG from EEG
        ECG = pop_select(EEG, 'channel', 61);
        ECG.chanlocs(end).labels = 'ecg_channel';
        EEG = pop_select(EEG, 'nochannel', 61);
        EEG.chanlocs(29).labels = 'TP9';
        EEG = pop_chanedit(EEG, 'lookup', channel_location_file);
        EEG.chanlocs_original = EEG.chanlocs;

        % Add dropdown electrode to ECG struct
        ECG.data(end + 1, :) = squeeze(EEG.data(29, :));
        ECG.nbchan = size(ECG.data, 1);
        ECG.chanlocs(end + 1).labels = 'dropdown_channel';
        EEG = pop_select(EEG, 'nochannel', 29);

        % Create a cardiac cycle time series
        r_lats = cell2mat({ECG.event(strcmpi({ECG.event.type}, 'R  1')).latency});
        r_cycle = NaN(size(ECG.times));
        r_lastrlat = NaN(size(ECG.times));
        current_r = 0;
        last_r = 0;
        t0 = 0;
        for t = 1 : ECG.pnts
            t0 = t0 + 1;
            if ismember(t, r_lats)
                t0 = 0;
                last_r = current_r;
                current_r = t;
                r_cycle(last_r + 1 : current_r) = linspace(0, 100, current_r - last_r);
            end
            r_lastrlat(t) = t0;
        end
        ECG.data(end + 1, :) = r_cycle;
        ECG.nbchan = size(ECG.data, 1);
        ECG.chanlocs(end + 1).labels = 'cycle';
        ECG.data(end + 1, :) = r_lastrlat;
        ECG.nbchan = size(ECG.data, 1);
        ECG.chanlocs(end + 1).labels = 'lastRlat';

        % Save ecg continuous data
        ECG = pop_saveset(ECG, 'filename', [subject '_ecg.set'], 'filepath', PATH_ECG, 'check', 'on', 'savemode', 'twofiles');

        % Remove noisy EEG channels
        [EEG, i1] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 10, 'norm', 'on', 'measure', 'kurt');
        [EEG, i2] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'prob');
        EEG.chans_rejected = horzcat(i1, i2);

        % Reref EEG to common average reference
        EEG = pop_reref(EEG, []);

        % Resample and filter for ICA
        ICA = pop_resample(EEG, 200);
        ICA = pop_basicfilter(ICA, [1 : EEG.nbchan], 'Cutoff', [1, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary');

        % Dummy epoch dataset for rejecting bad epochs before ICA
        ICA = eeg_regepochs(ICA, 'recurrence', 2, 'extractepochs', 'off');
        ICA = pop_epoch(ICA, {'X'}, [0, 2], 'newname', [num2str(id) '_dummy'], 'epochinfo', 'yes');
        ICA = eeg_checkset(ICA, 'eventconsistency');

        % Detect and remove artifacts
        ICA = pop_autorej(ICA, 'nogui', 'on', 'threshold', 1000, 'startprob', 5, 'maxrej', 5, 'eegplot', 'off');
  
        % Run ICA on random sample of 1000 trials
        n_trials_ica = 1000;
        if size(ICA.data, 3) > n_trials_ica
            idx = randsample([1 : size(ICA.data, 3)], 1000);
        else
            idx = [1 : size(ICA.data, 3)];
        end
        ICA = pop_selectevent(ICA, 'epoch', idx, 'deleteevents', 'off', 'deleteepochs', 'on', 'invertepochs', 'off');
        ICA = pop_runica(ICA, 'extended', 1, 'interupt', 'on');

        % Run ICLabel
        ICA = iclabel(ICA);

        % Copy ICA and classification results to EEG dataset
        EEG = pop_editset(EEG, 'icachansind', 'ICA.icachansind', 'icaweights', 'ICA.icaweights', 'icasphere', 'ICA.icasphere');
        EEG.etc = ICA.etc;

        % Save all ICs continuous data
        EEG = pop_saveset(EEG, 'filename', [subject '_icset.set'], 'filepath', PATH_ICSET, 'check', 'on', 'savemode', 'twofiles');

        % Autoclean data (keeping ECG artifacts)
        IC_muscle   = find(EEG.etc.ic_classification.ICLabel.classifications(:, 2) > 0.3);
        IC_eye      = find(EEG.etc.ic_classification.ICLabel.classifications(:, 3) > 0.3);
        IC_heart    = find(EEG.etc.ic_classification.ICLabel.classifications(:, 4) > 0.1);
        IC_out = setdiff(union(IC_muscle, IC_eye), IC_heart);
        EEG = pop_subcomp(EEG, IC_out, 0);
        EEG.ICs_removed = IC_out;

        % Interpolate missing channels
        EEG = pop_interp(EEG, EEG.chanlocs_original, 'spherical');

        % Save autocleaned continuous data
        EEG = pop_saveset(EEG, 'filename', [subject '_autocleaned.set'], 'filepath', PATH_AUTOCLEANED, 'check', 'on', 'savemode', 'twofiles');

        % Epoch eeg training data (y-data) and remove artifacted epochs
        TRY = EEG;
        TRY.event = TRY.event_R_nolock;
        TRY = pop_epoch(TRY, {'R'}, [-1, 1], 'newname', [num2str(id) '_Rnolock'], 'epochinfo', 'yes');
        TRY = eeg_checkset(TRY, 'eventconsistency');
        [TRY, idx] = pop_autorej(TRY, 'nogui', 'on', 'threshold', 1000, 'startprob', 5, 'maxrej', 5, 'eegplot', 'off');

        % Epoch ecg training data (X-data) 
        TRX = ECG;
        TRX.event = TRX.event_R_nolock;
        TRX = pop_epoch(TRX, {'R'}, [-1, 1], 'newname', [num2str(id) '_Rnolock'], 'epochinfo', 'yes');
        TRX = eeg_checkset(TRX, 'eventconsistency');
        TRX = pop_select(TRX, 'notrial', idx);

        % Epoch eeg prediction data (y-data) and remove artifacted epochs
        PRY = EEG;
        PRY.event = PRY.event_stimlocked;
        PRY = pop_epoch(PRY, {'stim'}, [-1, 2], 'newname', [num2str(id) '_stim'], 'epochinfo', 'yes');
        PRY = eeg_checkset(PRY, 'eventconsistency');
        [PRY, idx] = pop_autorej(PRY, 'nogui', 'on', 'threshold', 1000, 'startprob', 5, 'maxrej', 5, 'eegplot', 'off');

        % Epoch ecg prediction data (X-data) 
        PRX = ECG;
        PRX.event = PRX.event_stimlocked;
        PRX = pop_epoch(PRX, {'stim'}, [-1, 2], 'newname', [num2str(id) '_stim'], 'epochinfo', 'yes');
        PRX = eeg_checkset(PRX, 'eventconsistency');
        PRX = pop_select(PRX, 'notrial', idx);

        % Save X-data
        train_X = TRX.data;
        predi_X = PRX.data;
        save([PATH_REGRESSION_DATA subject '_train_X'], 'train_X');
        save([PATH_REGRESSION_DATA subject '_predi_X'], 'predi_X');

        % Save y-data channelwise
        for ch = 1 : EEG.nbchan
            train_y = squeeze(TRY.data(ch, :, :));
            predi_y = squeeze(PRY.data(ch, :, :));
            save([PATH_REGRESSION_DATA subject '_chan' num2str(ch) '_train_y'], 'train_y');
            save([PATH_REGRESSION_DATA subject '_chan' num2str(ch) '_predi_y'], 'predi_y');
        end

        % Detect double epoch events
        to_drop = [];
        for epo = 1 : length(PRY.epoch)
            if length(PRY.epoch(epo).event) > 1
                to_drop(end + 1) = PRY.epoch(epo).event(2);
            end
        end
        trialinfo = PRY.event;
        trialinfo(to_drop) = [];

        % Save trialinfo
        save([PATH_META subject '_trialinfo'], 'trialinfo');

        % Save chanlocs and time
        chanlocs = EEG.chanlocs;
        save([PATH_META 'chanlocs'], 'chanlocs');
        time_rlock = TRY.times;
        save([PATH_META 'time_rlock'], 'time_rlock');
        time_stimlock = PRY.times;
        save([PATH_META 'time_stimlock'], 'time_stimlock');

    end % End subject iteration

end % End part1

% ======================== PART2: Preprocessing statistics and save a less complex trialinfo for evaluation purposes =========================
if ismember('part2', to_execute)

    % Init eeglab and find chanlocfile
    addpath(PATH_EEGLAB);
    eeglab;

    % Iterating subject list
    n_chans_rejected = [];
    n_ICs_rejected = [];
    n_trials_r_locked_original = [];
    n_trials_r_locked_after_cleaning = [];
    n_trials_stim_locked_original = [];
    n_trials_stim_locked_after_cleaning = [];
    for s = 1 : length(subject_list)

        % Dataset name
        subject = subject_list{s}; 
        id = str2num(subject(3 : 4));

        % Load autocleaned data
        EEG = pop_loadset('filename', [subject '_autocleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

        % Get number of original r peak segments
        TRY = EEG;
        TRY.event = TRY.event_R_nolock;
        TRY = pop_epoch(TRY, {'R'}, [-1, 1], 'newname', [num2str(id) '_Rnolock'], 'epochinfo', 'yes');
        n_trials_r_locked_original(s) = size(TRY.data, 3); 

        % Get number of original stim segments
        PRY = EEG;
        PRY.event = PRY.event_stimlocked;
        PRY = pop_epoch(PRY, {'stim'}, [-1, 2], 'newname', [num2str(id) '_stim'], 'epochinfo', 'yes');
        n_trials_stim_locked_original(s) = size(PRY.data, 3); 

        % Get number of r peak segments in cleaned data
        load([PATH_REGRESSION_DATA subject '_chan' num2str(1) '_train_y']);
        n_trials_r_locked_after_cleaning(s) = size(train_y, 2);

        % Get number of stimlocked segments in cleaned data
        load([PATH_REGRESSION_DATA subject '_chan' num2str(1) '_predi_y']);
        n_trials_stim_locked_after_cleaning(s) = size(predi_y, 2);
        
        % Simplify trialinfo
        load([PATH_META, subject, '_trialinfo']);
        tinf = [];
        for e = 1 : length(trialinfo)
            if strcmpi(trialinfo(e).blockHR, 'SYS')
                tinf(e, 1) = 1;
            else 
                tinf(e, 1) = 2;
            end
            if strcmpi(trialinfo(e).corresp, 'c')
                tinf(e, 2) = 1;
            else 
                tinf(e, 2) = 2;
            end
        end
        save([PATH_META subject '_tinf'], 'tinf');

        n_chans_rejected(s) = length(EEG.chans_rejected);
        n_ICs_rejected(s) = length(EEG.ICs_removed);

    end % End subject iteration

    % Determining number of rejected epochs
    n_trials_r_rejected = n_trials_r_locked_original - n_trials_r_locked_after_cleaning;
    n_trials_stim_rejected = n_trials_stim_locked_original - n_trials_stim_locked_after_cleaning;

    % Concatenate data for readability
    col1 = [mean(n_chans_rejected),...
        mean(n_ICs_rejected),...
        mean(n_trials_r_locked_original),...
        mean(n_trials_r_locked_after_cleaning),...
        mean(n_trials_r_rejected),...
        mean(n_trials_stim_locked_original),...
        mean(n_trials_stim_locked_after_cleaning),...
        mean(n_trials_stim_rejected)];

    col2 = [std(n_chans_rejected),...
        std(n_ICs_rejected),...
        std(n_trials_r_locked_original),...
        std(n_trials_r_locked_after_cleaning),...
        std(n_trials_r_rejected),...
        std(n_trials_stim_locked_original),...
        std(n_trials_stim_locked_after_cleaning),...
        std(n_trials_stim_rejected)];
        
    preprostats = [col1', col2'];

end % End part2