function[EEG] = cfa_force2resp(EEG, subject, outpath, forcechans, chanlabels, etypes, maxrt, varargin)

	%
	%
	% WHAT THIS FUNCTION DOES:
	% ------------------------
	%
    % 1. For each force channel, determine threshold value for response detection
	% 2. Detect responses based on this value
	% 3. Add fields for each channels to event structure for storing RTs. Name of field corresponds channel label.
	%
	%
	% USAGE: EEG = cfa_force2resp(EEG, 'VP77', '/home/bestfolderever/', [65, 66], {'left', 'right'},...
	%                        {{'S666', 'S007'}, {'S911'}}, [1500])
	%
	%        EEG = cfa_force2resp(EEG, 'VP77', '/home/bestfolderever/', [65, 66], {'left', 'right'},...
	%                        {{'S666', 'S007'}, {'S911'}}, [1500], 'minresplatency', 200,...
	%                        'blwin', [-500, -300], 'filtbounds', [1, 20], 'percoutliers', 5, 'percofmax', 10);
	%
	%
	% INPUT ARGUMENTS:
	% ----------------
	%
	% needed: 
	% EEG              : The EEG structure
	% subject	       : Subject identifier as a string
	% outpath	       : A path for saving the output gfx into...
	% forcechans       : An array of integer values indicating which channels are force channels
	% chanlabels       : A cell array of strings, containing the labels of the force channels (same order as above)
	% etypes	       : A cell array of cells containing the names of all imperative events sorted by channel
	% maxrt	           : A integer indicating the max rt for the events
	%				
	% optional: 
	% 'minresplatency' : A value for the minimum latency to consider when determining valid responses, the lower
	%                    limit of the response window. Default is zero (0).
	% 'blwin'          : The baseline window can be customized. Default is [-200 0].
	% 'filtbounds'     : Lower and upper boundaries for bandpass filtering the force channels.
	%                    Default is [0.1, 40].
	% 'percoutliers'   : Percentage of highest force values, not considered for max force determination.
	%                    Default is 10.
    % 'percofmax'      : Percentage of determined maximum force needed to be a trial considered as a valid response.
    %					 Default is 20.
	%
	%
	% OUTPUT ARGUMENTS:
	% -----------------
	%	
	% EEG          : Modified EEG structure
	%
	% ------------------------
	% Stefan Arnau, 05.07.2017
	% ------------------------

	% Check if any input args
	if nargin < 7
	        error('Not enough input arguments... :)');
	        return;
	end

	% Check plausability of forcechan indices and forcechan labels
	if length(forcechans) ~= length(chanlabels)
		error('Number of force channels must be equal to force channel labels... :)');
		return;
	end

	% Calc latency modifiers
	latmod = 1000 / EEG.srate;

	% Init input parser
	p = inputParser;

	% Set Defaults
	default_minresplatency = 0;
	default_blwin = [-200, 0];
	default_filtbounds = [0.1, 40];
	default_percoutliers = 10;
	default_percofmax = 20;

	% Parsing inputs
	p.FunctionName  = mfilename;
	p.CaseSensitive = false;
	p.addRequired('EEG', @isstruct);
	p.addRequired('subject', @ischar);
	p.addRequired('outpath', @ischar);
	p.addRequired('forcechans', @isnumeric);
	p.addRequired('chanlabels', @iscellstr);
	p.addRequired('etypes', @iscell);
	p.addRequired('maxrt', @isnumeric);
	p.addParamValue('minresplatency', default_minresplatency, @isnumeric);
	p.addParamValue('blwin', default_blwin, @isnumeric);
	p.addParamValue('filtbounds', default_filtbounds, @isnumeric);
	p.addParamValue('percoutliers', default_percoutliers, @isnumeric);
	p.addParamValue('percofmax', default_percofmax, @isnumeric);
	parse(p, EEG, subject, outpath, forcechans, chanlabels, etypes, maxrt, varargin{:});

	% Check lengths of baseline win vector and filter boundaries
	if length(p.Results.filtbounds) ~= 2
		error('Please state exactly 2 values as lower and upper boundary for band pass filter force chans... :)');
		return;
	elseif p.Results.filtbounds(1) >= p.Results.filtbounds(2)
		error('Lower boundary of filtbound must be lower than upper boundary... :)');
		return;
	elseif length(p.Results.blwin) ~= 2
		error('Please state exactly 2 values defining the baseline... :)');
		return;
	elseif p.Results.blwin(1) >= p.Results.blwin(2)
		error('Lower boundary of blwin must be lower than upper boundary... :)');
		return;
	end
	
	% Filter force channels
    EEG  = pop_basicfilter(EEG, p.Results.forcechans, 'Cutoff', p.Results.filtbounds, 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary');

    % Create allstim field
    alltypes = {};
    for e = 1 : length(etypes)
    	alltypes(end + 1 : end + length(etypes{e})) = etypes{e};
    end

	% Iterate force channels
	for chan = 1 : length(p.Results.forcechans)
		
		% Determine stims where this chan is the correct response
		ctypes = etypes{chan};

		% Determine a max force value for events where this chan is correct response
		cutoff = [];
		for e = 1 : length(EEG.event)
			if ismember(EEG.event(e).type, ctypes)
				blmean = mean(EEG.data(p.Results.forcechans(chan), EEG.event(e).latency + p.Results.blwin(1) / latmod : EEG.event(e).latency + p.Results.blwin(2) / latmod));
				cutoff(end + 1) = max(EEG.data(p.Results.forcechans(chan), EEG.event(e).latency + p.Results.minresplatency / latmod : EEG.event(e).latency + maxrt / latmod) - blmean);
			end
		end

		% Check if valid events were found
		if length(cutoff) < 1
			error('No events found... :)');
			return;
		end

		% Determine cutoff value for response detection in current channel
		cutoff = sort(cutoff); % Sort cutoff
		cutoff = cutoff(length(cutoff) - ceil(length(cutoff) * (1 / (100 / p.Results.percoutliers)))); % Remove x-percent as outliers and determine max value
		maxforce = cutoff; % Save maxforce	
		cutoff = cutoff * (1 / (100 / p.Results.percofmax)); % Set cutoff at x percent of actual max value (after outliers removal)

		% Iterate all stimulus events and get baseline and max force values and add lats to event
		maxresps = [];
		blforce = [];
		for e = 1 : length(EEG.event)
			if ismember(EEG.event(e).type, alltypes)

				% Determine and collect bl-mean
				blmean = mean(EEG.data(p.Results.forcechans(chan), EEG.event(e).latency + p.Results.blwin(1) / latmod : EEG.event(e).latency + p.Results.blwin(2) / latmod));
				blforce(end + 1) = blmean;

				% Get response window data
				respdata = EEG.data(p.Results.forcechans(chan), EEG.event(e).latency + p.Results.minresplatency / latmod : EEG.event(e).latency + maxrt / latmod) - blmean;

				% Get val and idx of max response value and save value to maxresps
				[maxval, maxidx] = max(respdata);
				maxresps(end + 1) = maxval;

				% If maxval exceeds cutoff add rt to event
				if maxval >= cutoff

					% loop for index in response window exceeding cutoff
					idx = 1;
					while respdata(idx) < cutoff
						idx = idx + 1;
					end

					fn = p.Results.chanlabels{chan};
					rt = (p.Results.minresplatency / latmod + idx) * latmod; % Calc rt
					EEG.event(e).(fn) = rt; % Add rt to corresponding stimulus event

					% Add force detection event event
					EEG.event(end + 1) = EEG.event(e);
					EEG.event(end).latency = EEG.event(e).latency + rt;
					EEG.event(end).code = p.Results.chanlabels{chan};
					EEG.event(end).type = p.Results.chanlabels{chan};
            		EEG.event(end).(fn) = rt;
            		EEG.event(end).duration = 1;
				end
			end
		end

		% Sort force amp for all stimulus events for distribution plotting
		maxsort = sort(maxresps); % Sort maxresps

		% Find cutoff indices for zeroval, cutoffval and maxval (outlierdetection)
		[zeroval, zeroidx] = closest(maxsort, 0);
		[cutoffval, cutoffidx] = closest(maxsort, cutoff);
		[maxval, maxidx] = closest(maxsort, maxforce);

		% Plot stuff
		f = figure('Visible','off');

		% Subplot 1: Response force distribution
		subplot(2, 2, 1);
		plot(maxsort, 'c', 'LineWidth', 1.5);
		xlim([1, length(maxsort)]);
		maxy = max(maxsort) + max(maxsort) / 20;
		miny = min(maxsort) - max(maxsort) / 20;
		ylim([miny, maxy]);
		title(['response force distribution ' p.Results.subject ' - channel ' int2str(p.Results.forcechans(chan)) ' (' p.Results.chanlabels{chan} ')']);
		set(gca, 'XTick', [zeroidx, cutoffidx, length(maxsort)]);
		xlabel(['events: ' int2str(length(maxsort)) ', responses: ' int2str(length(maxsort) - (cutoffidx - 1)) ', < bl: ' int2str(zeroidx - 1)]);
		grid on;
		hold on;

		plot([zeroidx, zeroidx],[miny, maxy], 'g'); % Plot vertical line at baseline force

		plot([cutoffidx, cutoffidx],[miny, maxy], '-m'); % Plot vertical line at cutoff
		plot([1, length(maxsort)],[cutoff, cutoff], '-m'); % Plot horizontal line at cutoff

		plot([1, length(maxsort)],[maxval, maxval], '--k'); % Plot horizontal line at zeroidx
		plot([maxidx, maxidx],[miny, maxy], '--k'); % Plot vertical line at maxval
		hold off;

		subplot(2, 2, 2);
		title('distribution of detected responses');
		piedata = [length(maxsort) - (cutoffidx - 1), length(maxsort) - (length(maxsort) - (cutoffidx - 1)) - (zeroidx - 1), zeroidx - 1];
		pielabels = {'valid', 'below threshold', 'below baseline'};
		pie(piedata);
		colormap([0, 1, 1; 1, 0, 1; 0, 1, 0]);
		legend(pielabels, 'Location', 'southoutside', 'Orientation', 'vertical');

		subplot(2, 2, [3, 4]);
		plot(maxresps, 'c', 'LineWidth', 1.5);
		xlim([1, length(maxresps)]);
		maxy = max(maxresps) + max(maxresps) / 20;
		miny = min(maxresps) - max(maxresps) / 20;
		ylim([miny, maxy]);
		title(['response / baseline force all trials']);
		set(gca, 'XTick', [ceil(length(maxsort)) * 0.25, ceil(length(maxsort)) * 0.5, ceil(length(maxsort)) * 0.75, length(maxsort)]);
		set(gca, 'YTick', [0, cutoff]);
		xlabel('imperative stimulus events');
		ylabel('cutoff value');
		grid on;
		hold on;
		plot([1 , length(maxresps)], [cutoff, cutoff], '-m', 'LineWidth', 1.5); % Plot cutoff line
		plot(maxresps, 'k+'); % Plot markers for responses
		plot(blforce, 'g');
		hold off;
	
		% Save the plot as png
		print(f, '-dpng', '-r300', [p.Results.outpath p.Results.subject '_channel' int2str(p.Results.forcechans(chan)) '_' p.Results.chanlabels{chan}]); 

	end % End channel iteration

	% Check events
	EEG = eeg_checkset(EEG, 'eventconsistency');
end