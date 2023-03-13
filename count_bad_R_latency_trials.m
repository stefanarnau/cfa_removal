% Clean up residuals
clear all;

% Path vars
PATH_RAW    = '/mnt/data_fast/simHR/eeg/0_raw/';
PATH_EEGLAB = '/home/plkn/eeglab2022.1/';
PATH_FORCE  = '/home/plkn/Desktop/force_dump/';
% Subject list
subject_list = {'VP02', 'VP03', 'VP04', 'VP05', 'VP06', 'VP07', 'VP08', 'VP09', 'VP10', 'VP11',...
                'VP12', 'VP13', 'VP14', 'VP15', 'VP16', 'VP17', 'VP18', 'VP19', 'VP20', 'VP21',...
                'VP22', 'VP23', 'VP24', 'VP25', 'VP26', 'VP27', 'VP28', 'VP29', 'VP30', 'VP31',...
                'VP32', 'VP33', 'VP34', 'VP35', 'VP36', 'VP37', 'VP38', 'VP39', 'VP40', 'VP41'};

% Init EEGLab
addpath(PATH_EEGLAB);
eeglab;

bad_latency_stats = [];

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
    [EEG.event_stimlocked, bad_latency_stats(s, 1), bad_latency_stats(s, 2)] = get_stim_events(EEG, id, PATH_FORCE, [61, 62]);

end

% Add percentages
bad_latency_stats(:, 3) = bad_latency_stats(:, 2) ./ (bad_latency_stats(:, 1) / 100);

% Get means
mean(bad_latency_stats, 1)

% Get standard deviations
std(bad_latency_stats, [], 1)



