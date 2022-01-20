function[S] = get_R_nolock_events(EEG)
 
    c = 0;
    r = 0;
    S = struct('latency', {},...
            'code', {},...
            'type', {},...
            'Rnum', {},...
            'urevent', {},...
            'duration', {}...                     
            );
        
    % Iterate events
    for e = 2 : length(EEG.event) - 1

        % Only non locking R peaks
        if strcmpi(EEG.event(e).type, 'R  1') &...
           ~strcmpi(EEG.event(e - 1).type(1), 'S') &...
            strcmpi(EEG.event(e + 1).type(1), 'R')
            
            c = c + 1;
            r = r + 1;
            S(c).latency = EEG.event(e).latency;
            S(c).code = 'R';
            S(c).type = 'R';
            S(c).Rnum = r; 
            S(c).urevent = c;
            S(c).duration = 1;           
        elseif strcmpi(EEG.event(e).type, 'boundary')
            c = c + 1;
            S(c).latency = EEG.event(e).latency;
            S(c).code = 'boundary';
            S(c).type = 'boundary';
            S(c).Rnum = NaN; 
            S(c).urevent = c;
            S(c).duration = 1;
        end
    end
end