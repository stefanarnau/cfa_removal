function[S] = get_stim_events(EEG, id, forcepath, forcechans)

    % Define crrect response stimlabels
    stimlabsl = {};
    stimlabsr = {};
    for e = 1 : length(EEG.event)
        if strcmpi(EEG.event(e).type(1), 'S')
            enum = str2num(EEG.event(e).type(2 : end));
            if enum < 161
                if enum < 81
                    stimid = 'X';
                else
                    stimid = 'S';
                end
                if mod(enum, 20) > 0 & mod(enum, 20) < 11
                    mappedleft = 'X';
                else
                    mappedleft = 'S';
                end
                if strcmpi(mappedleft, stimid)
                    stimlabsl{end + 1} = EEG.event(e).type;
                else
                    stimlabsr{end + 1} = EEG.event(e).type;
                end
            end
        end
    end

    % Condense stimlabs for force detection
    stimlabsl = unique(stimlabsl);
    stimlabsr = unique(stimlabsr);

    % Read force
    EEG = cfa_force2resp(EEG, num2str(id), forcepath, forcechans, {'left', 'right'}, {stimlabsl, stimlabsr}, 1500, 'percofmax', 10, 'minresplatency', 100, 'blwin', [-200, 0]);

    % Create event structure S for stimulus locked signal analysis
    c = 0;
    bc = 0;
    bl = 0;
    obl = 0;
    blockmaxlat = zeros(1, 5);
    S = struct('latency', {},...
            'code', {},...
            'type', {},...
            'stimnum', {},...
            'blocknum', {},...
            'blockstimnum', {},...
            'blocktot', {},...
            'blockperc', {},...
            'stimid', {},...
            'stimloc', {},... 
            'mappedleft', {},... 
            'correctresp', {},...   
            'corresp', {},...            
            'startHR', {},...                          
            'blockHR', {},...
            'rlat', {},...
            'targetlat', {},...
            'laterror', {},...
            'latexclude', {},...         
            'response', {},...  
            'rt', {},...
            'acc', {},...  
            'urevent', {},...
            'duration', {}...                     
            );
    
    % Iterate events
    for e = 2 : length(EEG.event) - 1

        % If stimulus event
        if strcmpi(EEG.event(e).type(1), 'S')

            % Get code
            enum = str2num(EEG.event(e).type(2 : end));

            % Loop for rpeak latency
            f = e;
            while ~strcmpi(EEG.event(f).type, 'R  1') & f > 1
                f = f - 1;
            end

            if strcmpi(EEG.event(f).type, 'R  1')
                rlat = EEG.event(e).latency - EEG.event(f).latency;
            else
                rlat = NaN;
            end

            if enum < 161
                c = c + 1;
                bc = bc + 1;
                bl = mod(enum, 5);
                if mod(enum, 5) == 0
                    bl = 5;
                end
                if bl > obl
                    obl = bl;
                    bc = 1;
                    reflat = EEG.event(e).latency;
                end
                S(c).latency = EEG.event(e).latency;
                S(c).code = enum;
                S(c).type = 'stim';
                S(c).stimnum = c;
                S(c).blocknum = bl;
                S(c).blockstimnum = bc;
                S(c).blocktot = EEG.event(e).latency - reflat;
                if EEG.event(e).latency - reflat > blockmaxlat(bl)
                    blockmaxlat(bl) = EEG.event(e).latency - reflat; % Detect latency of last stim event in block
                end
                if enum < 81
                    S(c).stimid = 'X';
                else
                    S(c).stimid = 'S';
                end
                if enum < 41 | (enum > 80 & enum < 121)
                    S(c).stimloc = 'l';
                else
                    S(c).stimloc = 'r';
                end
                if mod(enum, 20) > 0 & mod(enum, 20) < 11
                    S(c).mappedleft = 'X';
                else
                    S(c).mappedleft = 'S';
                end
                if strcmpi(S(c).mappedleft, S(c).stimid)
                    S(c).correctresp = 'l';
                else
                    S(c).correctresp = 'r';
                end
                if strcmpi(S(c).stimloc, S(c).correctresp)
                    S(c).corresp = 'c';
                else
                    S(c).corresp = 'nc';
                end
                if mod(enum, 40) > 0 & mod(enum, 40) < 21
                    S(c).startHR = 'SYS';
                else
                    S(c).startHR = 'DIA';
                end
                if mod(enum, 10) > 0 & mod(enum, 10) < 6
                    S(c).blockHR = 'SYS';
                else
                    S(c).blockHR = 'DIA';
                end

                % Trigger latencies
                S(c).rlat = rlat;
                if S(c).blocknum == 1
                    S(c).targetlat = NaN;
                elseif strcmpi(S(c).blockHR, 'SYS')
                    S(c).targetlat = 230;
                elseif strcmpi(S(c).blockHR, 'DIA')
                    S(c).targetlat = 530;
                end
                if S(c).blocknum == 1
                    S(c).laterror = NaN;
                elseif S(c).rlat == NaN
                    S(c).laterror = NaN;
                else
                    S(c).laterror = S(c).rlat - S(c).targetlat;
                end
                if S(c).laterror == NaN
                    S(c).latexclude = 1;
                elseif abs(S(c).laterror) <= 30
                    S(c).latexclude = 0;
                else
                    S(c).latexclude = 1;
                end

                % Behavior
                if EEG.event(e).left & EEG.event(e).right
                    if EEG.event(e).left <= EEG.event(e).right
                        S(c).response = 'l';
                        S(c).rt = EEG.event(e).left;
                    else
                        S(c).response = 'r';
                        S(c).rt = EEG.event(e).right;
                    end
                elseif EEG.event(e).left
                    S(c).response = 'l';
                    S(c).rt = EEG.event(e).left;
                elseif EEG.event(e).right
                    S(c).response = 'r';
                    S(c).rt = EEG.event(e).right;
                else
                    S(c).response = 'NONE';
                    S(c).rt = NaN;
                end
                if strcmpi(S(c).response, 'NONE')
                    S(c).acc = 2;
                elseif strcmpi(S(c).response, S(c).correctresp)
                    S(c).acc = 1;
                else
                    S(c).acc = 0;
                end
                S(c).urevent = c;
                S(c).duration = 1;

            end % End '< 161' check

        elseif strcmpi(EEG.event(e).type, 'boundary')
            c = c + 1;
            S(c).latency = EEG.event(e).latency;
            S(c).code = 'boundary';
            S(c).type = 'boundary';
            S(c).urevent = c;
            S(c).duration = 1;
        end 
    end % End event loop

    % Add percentage of block info
    for c = 1 : length(S)
        if strcmpi(S(c).type, 'stim')
            S(c).blockperc = S(c).blocktot / (blockmaxlat(S(c).blocknum) / 100); 
        else
            S(c).blockperc = NaN;
        end
    end

    % Remove untolerable latency trials (latency offset Rpeak <-> stimulus)
    todrop = [];
    for c = 1 : numel(S)
        if strcmpi(S(c).type, 'stim') & S(c).latexclude == 1
            todrop(end + 1) = c;
        end
    end
    S(todrop) = [];
end