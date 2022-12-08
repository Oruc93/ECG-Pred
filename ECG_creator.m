function [t,ecg,beat_begin,beat_type] = ECG_creator(BBI,samplerate,blend_epsilon)

%ECG_creator creates ECG timeseries from a BBI-vector, using sample
%heartbeats from Beat_library and setting them after each other. The
%baseline is set to zero. Ectopic beats (for this purpose define as beats
%which have P - wave elements within T - wave parts of the former BBI) are
%directly excluded
%   INPUT:
%   BBI = vector containing BBI data in ms
%   samplerate = samplerate of the output signal (1000 Hz is advised,
%   otherwise integer multipliers of 1000 Hz are advisable to avoid rounding
%   errors.)
%   blend_epsilon = boundary value of blending. When the ECG at the end or
%   beginning overlap, the first boundary crossing sample defines the start
%   of overblending. Default = 0;
%   OUTPUT:
%   ecg = timeseries of modelled ECG
%   t = time vector of modelled ECG
%   beat_begin = time of the begining of each included beat. Excluded beats
%   are depicted by nan's
%   beat_type = vector informing over the type of beat use: 0 = beat
%   from library, 1 = beat from library with fading at the begining to the
%   end of the beat before, 2 = no beat, because beats are to close


% Load sample data
file='ECG_library.mat';
ECG_library=load(fullfile(file)); % Load library
sample_samplerate=ECG_library.samplerate; % Sample rate of the library

% Set samplerate to Library samplerate if not given
if nargin < 2
   samplerate=sample_samplerate;
end
% Epsilon area around zero for over fading between beats:
if nargin < 3
    blend_epsilon=0; %0.5*10^(-3);
end

% Output ECG features
R_positions=cumsum(floor((BBI/1000)*samplerate)); % positions of R-peaks
% size(R_positions)
number_of_beats=length(BBI); % Count heartbeats
number_of_samples=R_positions(end)+(ceil(3/5*((BBI(end)/1000)*samplerate)))+1; % estimate of length of ecg (3/5 is the factor typically used to distribute the ECG parts in the library

% Preset ECG-vector
ecg_dummy=zeros(number_of_samples,1); % zero baseline
beat_begin_sample=zeros(1,number_of_beats); % vector containing beat starts
beat_type=zeros(1,number_of_beats); % vector with beat types

% Set zero parameters
effective_end=0; 
fadecheck_old=0;


% Extract BBI parameters
BBI_quantization=ECG_library.BBI(2)-ECG_library.BBI(1); % qunatization of BBI
half_quantization=floor(BBI_quantization/2); 
minimal_BBI=ECG_library.BBI(1)-half_quantization; % Minimal still accepted BBI
maximal_BBI=ECG_library.BBI(end)+half_quantization; % Maximal still accepted BBI
% fprintf(strcat('\n Minimal BBI ', string(minimal_BBI))) % Analyse
% occuring BBIs
% fprintf(strcat('\n Maximal BBI ', string(maximal_BBI)))
% Inserting ECG-samples
for ii = 1:number_of_beats
    % fprintf(strcat('\n BBI ', string(BBI(ii))))
    if BBI(ii)>(minimal_BBI) && BBI(ii)<(maximal_BBI) % Limits of the library. Out of limit beats are depicted as zeros times
        sample_BBI=round(BBI(ii)/BBI_quantization)*BBI_quantization; % choose BBI of sample ecg beat (half quantization in each direction is included)
        sample_index=(sample_BBI-(ECG_library.BBI(1)-BBI_quantization))/BBI_quantization; % Index of the sample within array
        sample=ECG_library.(strcat('BBI_',num2str(sample_BBI))); % Get the sample
        
        
        % Include data
        sample_length=length(sample); % Length of the sample
        sample_start=R_positions(ii)-(ECG_library.samples_before_R(sample_index)); % starting position of sample within ecg
        sample_end=sample_start+sample_length-1; % ending position of sample within ecg
        effective_start=(sample_start+ECG_library.smoothing_edges(sample_index,1))-1; % First non-zero sample
        first_peak_position=sample_start+(ECG_library.peak_position_best_fit(1,sample_index))-1; % Position of the P peak
        % Fading conditions - PART ONE
        % Figure out last zero crossing before peak of new sample
        zero_indices=find(sample<=blend_epsilon);
        relevant_zero=find(zero_indices<first_peak_position-sample_start);
        epsilon_cross_sample_new=zero_indices(relevant_zero(end)); % Index within sample where epsilon is crossed
        fadecheck_new= sample_start+epsilon_cross_sample_new-1;

        
        % Check for overlap
        if effective_start>=effective_end % NO OVERLAP
            ecg_dummy(sample_start:sample_end)=ecg_dummy(sample_start:sample_end)+sample; % Include Sample
            % Relevant features for the next overlap situation to be kept from
            % old sample
            effective_end=(sample_start+ECG_library.smoothing_edges(sample_index,2))-1; % Last non-zero sample
            last_peak_position_before=sample_start+(ECG_library.peak_position_best_fit(3,sample_index))-1; % Position of the T peak
%             figure;plot(ecg_dummy);hold on; plot(sample)
            sample_index_before=sample_index;
            sample_before=sample; 
            % Fading conditions - PART TWO
            % Figure out last zero crossing after peak of old sample
            zero_indices_old=find(sample_before<=blend_epsilon);
            relevant_zero_old=find(sample_start+zero_indices_old>last_peak_position_before);
%             sample_start_before=sample_start;
            epsilon_cross_sample_old=zero_indices_old(relevant_zero_old(1))-1; % Index within sample where epsilon is crossed
            fadecheck_old = sample_start+epsilon_cross_sample_old;
            beat_begin_sample(ii)=sample_start;
            
        elseif last_peak_position_before>first_peak_position % OVERLAP OF PEAKS
            warning('Beats to close - T and P wave overlap')
            beat_begin_sample(ii)=nan;
            beat_type(ii)=2;
            % fprintf('Peak overlap \n')
            
        elseif fadecheck_new<fadecheck_old % Exclude ectopic beats (skip next beat if anything within last beat peak and first zero crossing before or after is touched)
            beat_begin_sample(ii)=nan;
            beat_type(ii)=2;

            fprintf('sample skip \n')            
            continue;
        else % ( Overblending between beats)
            spline_start=fadecheck_old;
            spline_end=fadecheck_new;
            steps_blend=spline_start:spline_end;
            blend_weight=linspace(1,0,length(steps_blend))'; % Weights for the first ecg
            blend_end_old=min([length(sample_before), epsilon_cross_sample_old+length(steps_blend)-1]);
            blend_start_new=max([1 (epsilon_cross_sample_new-length(steps_blend)+1)]);
            blend_part=zeros(1,length(steps_blend))';
            blend_size_before=(blend_end_old-epsilon_cross_sample_old);
            blend_size_after=((epsilon_cross_sample_new-blend_start_new));
            blend_part(1:blend_size_before)=blend_part(1:(blend_end_old-epsilon_cross_sample_old))+blend_weight(1:blend_size_before).*sample_before(epsilon_cross_sample_old+1:blend_end_old);
            blend_part((end-blend_size_after):length(steps_blend))=blend_part((end-blend_size_after):length(steps_blend))+(1-blend_weight(end-blend_size_after:end)).*sample(blend_start_new:epsilon_cross_sample_new);
            rest_part=sample(epsilon_cross_sample_new+1:end);
            
%         else % ( Overblending between beats by spline interpolation)
%             spline_start=last_peak_position_before+1;
%             spline_end=first_peak_position-1;
%             steps_spline=spline_start:spline_end;
%             x_before_spline=((spline_start-3):(spline_start-1)); % x values before spline
%             x_after_spline=((spline_end+1):spline_end+3); % x values after spline
%             x_middle=steps_spline(round(length(steps_spline)/2));
%             y_before_spline=sample_before((ECG_library.peak_position_best_fit(3,sample_index_before)-2):(ECG_library.peak_position_best_fit(3,sample_index_before))); % Values from front beat
%             y_after_spline=sample(ECG_library.peak_position_best_fit(1,sample_index):ECG_library.peak_position_best_fit(1,sample_index)+2); % Values from back beat
%             x_spline=[x_before_spline x_middle x_after_spline]; % Take boundary terms, plus additionally three values (x-values)
%             y_spline=[y_before_spline' 0 y_after_spline']; % y-values
%         
%             % Interpolation
%             spline=interp1(x_spline,y_spline,steps_spline,'spline');
%             
%             blend_part=spline; % Blended part
%             rest_part=sample(ECG_library.peak_position_best_fit(1,sample_index):end); % Unblended part
%           

%         else % ( Overblending between beats by spline interpolation)
%             spline_start=fadecheck_old-3;
%             spline_end=fadecheck_new+3;
%             steps_spline=spline_start+3:spline_end-3;
%             x_before_spline=((spline_start):(spline_start+3)); % x values before spline
%             x_after_spline=((spline_end-3):spline_end); % x values after spline
%             y_before_spline=sample_before((epsilon_cross_sample_old-3):(epsilon_cross_sample_old)); % Values from front beat
%             y_after_spline=sample(epsilon_cross_sample_new:epsilon_cross_sample_new+3); % Values from back beat
%             x_spline=[x_before_spline x_after_spline]; % Take boundary terms, plus additionally three values (x-values)
%             y_spline=[y_before_spline' y_after_spline']; % y-values
%         
%             % Interpolation
%             spline_inter=interp1(x_spline,y_spline,steps_spline,'spline');
%             spline=[y_before_spline(1:3)' spline_inter y_after_spline(2:4)'];
%             
%             blend_part=spline; % Blended part            
%             rest_part=sample(epsilon_cross_sample_new+4:end);
%             
            % Include sample parts
            ecg_dummy(spline_end+1:sample_end)=rest_part; % Include unchanged sample
            ecg_dummy(spline_start:spline_end)=blend_part; % Include blended sample
            % fprintf('sample blending \n')
            
            beat_type(ii)=1;
            
            % Relevant features for the next overlap situation to be kept from
            % old sample
            effective_end=(sample_start+ECG_library.smoothing_edges(sample_index,2))-1; % Last non-zero sample
            last_peak_position_before=sample_start+(ECG_library.peak_position_best_fit(3,sample_index))-1; % Position of the T peak
            sample_index_before=sample_index;
            sample_before=sample; 
                    
            % Fading conditions - PART TWO
            % Figure out last zero crossing after peak of old sample
            zero_indices_old=find(sample_before<=blend_epsilon);
            relevant_zero_old=find(sample_start+zero_indices_old>last_peak_position_before);
            epsilon_cross_sample_old=zero_indices_old(relevant_zero_old(1))-1; % Index within sample where epsilon is crossed
            fadecheck_old = sample_start+epsilon_cross_sample_old;
            beat_begin_sample(ii)=sample_start;
            
        end
        

    else 
        beat_type(ii)=2;
        fprintf('No fitting beat in library \n')
    end
end

% Transform data into the right samplerate
if ~(samplerate==sample_samplerate)
    samplerate_factor=round(samplerate/sample_samplerate); % factor of upsampling
    ecg_dummy=interp(ecg_dummy,samplerate_factor); % Upsampling
end
% Output
ecg=ecg_dummy; % final ecg
t=((1:length(ecg))/samplerate)'; % Create time vactor corresponding to the ecg
beat_begin=beat_begin_sample./samplerate; % Vector containing beat begining times
end