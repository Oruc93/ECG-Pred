% This script constructs an ECG out of an BBI vector and multiple heartbeat
% snippets
% The BBI vector is derived from an ECG generated by ECGSYN
% The snippets of single heartbeats with varying RR-intervals are real
% heartbeats measured and cut by Fabian
% The final ECG is just lined up snippets with the BBI defining the
% RR-intervals. The method is coded by Fabian

% Additionaly we save the BBI, symbols, words and different non-linear
% parameters for the whole timeseries in a HDF5 file

% The reason we transform the ECG from ECGSYN to an BBI and back is that
% the generation of long ECGs with ECGSYN is prone to instabilities and
% contains artifacts
% The algorithm of Fabian has no major artifacts like that but loses some
% of the morphologies of ECGSYN generated ECGs (amplitude adjustment by
% respiration)

% Change the current folder to the folder of this m-file.
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end
clear all
delete data/*.bin
addpath('./tf')
addpath('./nl')
addpath('./data')
%% parameters for ECGSYN
% Fabians method was intended for BBIs between 400ms and 1600ms
% 150 beats per minute = 400ms average of BBI
% 37.5 beats per minute = 1600ms average of BBI
sfecg_syn = 256; % samplerate of resulting ecg
N = 1536; % 512 beats are a bit more than 8 minutes
Anoise = 0;
% Maximum Variability for Fabians method without beat skips
% hrmean = (150-37.5)/2+37.5;
% hrstd = 14; 
% Average human
%hrmean = 70;
hr_min = 50;
hr_max = 80;
hrstd = 6; % results in median HRV of about 50ms

%% parameters for Fabians method
sfecg = 1024; % samplerate of resulting ecg

amount = 1280; % amount of generated ECGs
ecg_duration = 18*60*sfecg; % duration of ECG interval in samples

%% Loop for Generation
for ii = 1:amount
    fprintf("\nGenerating ECG number %i\n", ii)
    check = true;
    lap = 0;
    while check % loop until a ECG with no beat skip is generated
        hrmean = (hr_max - hr_min)*rand(1,1) + hr_min;
        % generate ECG with ECGSYN
        [s, ipeaks] = ecgsyn(sfecg_syn,N,Anoise,hrmean,hrstd);
        r_peaks = ipeaks==3; % Isolate R-peaks
        r_peaks_index = find(r_peaks); % Extract indices of R-peaks
        r_peaks_ms = r_peaks_index/sfecg_syn*1000; % transform indices into ms
        BBI = r_peaks_ms(2:end) - r_peaks_ms(1:end-1); % calculate BBI in ms
        
        % Upsample ipeaks to same samplerate as Fabians Methode
        % by inserting zeros
        ipeaks = upsample(ipeaks,sfecg/sfecg_syn);

        % Fabians method
        % Bachelor thesis Fabian Chapter 2.3
        % Fit heartbeat snippets onto BBI vector
        % for each BBI value a snippet with same length is chosen
        % The chosen snippets then are lined up and interpolated in between
        [t,ecg,beat_begin,beat_type] = ECG_creator(BBI, sfecg);
        fprintf(strcat('ECG with Fabians method sampled with ', string(sfecg), 'Hz \n'))

        % Cutting N long interval of whole BBI timeseries
        BBI = BBI(1:N);

        % Checks if generation worked
        fprintf(strcat(string(sum(beat_type==2)), ' beats skipped\n'))
        if sum(beat_type==2) == 0
            check = false;
        else
            fprintf('At least one beat skipped. We try again\n')
        end
        if lap > 12
            error("Cannot generate ECG without skipping beats. Try " + ...
                "smaller hrstd")
        end
        lap = lap +1;
    end
    
    
    % calculate symbols and words
    [words, symbols] = calc_symboldynamics(BBI, 0.01, "movdiff");
    % calculate non-linear parameters
    forbword = forbidden_words(words);
    fwshannon_param = fwshannon(words);
    fwrenyi_025_param = fwrenyi(words, 0.25);
    fwrenyi_4_param = fwrenyi(words, 4);
    wpsum02_param = wpsum02(words);
    wpsum13_param = wpsum13(words);
    wsdvar_param = wsdvar(symbols);
    plvar_5_param = plvar(BBI, 5);
    plvar_10_param = plvar(BBI, 10);
    plvar_20_param = plvar(BBI, 20);
    phvar_20_param = phvar(BBI, 20);
    phvar_50_param = phvar(BBI, 50);
    phvar_100_param = phvar(BBI, 100);
    
    nlp = [
    forbword, fwshannon_param, fwrenyi_025_param, fwrenyi_4_param, ...
    wsdvar_param, wpsum02_param, wpsum13_param, plvar_5_param, ...
    plvar_10_param, plvar_20_param, phvar_20_param, phvar_50_param, ...
    phvar_100_param];

    nlp_name = [
    "forbword", "fwshannon", "fwrenyi 0.25", "fwrenyi 4", "wsdvar", ...
    "wpsum 02", "wpsum 13", "plvar 5", "plvar 10", "plvar 20", ...
    "phvar 20", "phvar 50", "phvar 100"];
    
    % Write data in bin format
    % 80% of the amount of generated ECG will be used for training
    % We seperate the sets early on to prevent data leakage
    if ii<=0.8*amount
        % Saving trainings set
        name_ECG = './data/ECG_training.bin';
        name_t = './data/t_training.bin';
        name_ipeaks = './data/ipeaks_training.bin';
        name_BBI = './data/BBI_training.bin'; % BBI array
        name_symbols = './data/symbols_training.bin'; % symbols array
        name_words = './data/words_training.bin'; % words array
        name_nlp = {};
        for n = 1:length(nlp_name)
            name_nlp(n) = {strcat('./data/', nlp_name(n), '_training.bin')}; % non-linear parameters
        end
    else
        % Saving test set
        name_ECG = './data/ECG_test.bin';
        name_t = './data/t_test.bin';
        name_ipeaks = './data/ipeaks_test.bin';
        name_BBI = './data/BBI_test.bin';
        name_symbols = './data/symbols_test.bin';
        name_words = './data/words_test.bin';
        name_nlp = {};
        for n = 1:length(nlp_name)
            name_nlp(n) = {strcat('./data/', nlp_name(n), '_test.bin')}; % non-linear parameters
        end
    end
    
    % Variablen speichern
    fileID = fopen(name_ECG,'a');
    fwrite(fileID, ecg(1:ecg_duration)', 'double');
    fclose(fileID);

    fileID = fopen(name_t,'a');
    fwrite(fileID, t(1:ecg_duration)', 'double');
    fclose(fileID);

    fileID = fopen(name_ipeaks,'a');
    fwrite(fileID, ipeaks(1:ecg_duration)', 'int');
    fclose(fileID);

    fileID = fopen(name_BBI,'a'); % opens with access type append
    fwrite(fileID, BBI', 'double');
    fclose(fileID);

    fileID = fopen(name_symbols,'a'); % opens with access type append
    fwrite(fileID, symbols, 'double');
    fclose(fileID);

    fileID = fopen(name_words,'a'); % opens with access type append
    fwrite(fileID, words, 'double');
    fclose(fileID);
    
    %loop und nlp speichern
    for n = 1:length(name_nlp)
        fileID = fopen(string(name_nlp(n)),'a'); % opens with access type append
        fwrite(fileID, nlp(n), 'double');
        fclose(fileID);
    end

end

% Transfer bin savefiles to hdf5 datasets and attributes
% training
% ecg
fileID = fopen('./data/ECG_training.bin','rb'); % opens with access type read
A = fread(fileID, [ecg_duration,0.8*amount], 'double'); % read file
fclose(fileID); % close file
delete data/training.h5
h5create("data/training.h5","/ECG", [ecg_duration,0.8*amount]) % create hdf5
h5write("data/training.h5","/ECG", A) % save ECGs in hdf5
% time
fileID = fopen('./data/t_training.bin','rb'); % opens with access type read
A = fread(fileID, [ecg_duration,0.8*amount], 'double'); % read file
fclose(fileID); % close file
h5create("data/training.h5","/time", [ecg_duration,0.8*amount]) % create hdf5
h5write("data/training.h5","/time", A) % save ECGs in hdf5
% peaks
% classification of waves, peaks and otherwise
fileID = fopen('./data/ipeaks_training.bin','rb'); % opens with access type read
A = fread(fileID, [ecg_duration,0.8*amount], 'int'); % read file
fclose(fileID); % close file
h5create("data/training.h5","/RP", [ecg_duration,0.8*amount]) % create hdf5
h5write("data/training.h5","/RP", A) % save classification in hdf5
% BBI
fileID = fopen('./data/BBI_training.bin','rb'); % opens with access type read
A = fread(fileID, [N, 0.8*amount], 'double'); % read file
fclose(fileID); % close file
h5create("data/training.h5","/BBI", [N, 0.8*amount]) % create hdf5
h5write("data/training.h5","/BBI", A) % save BBIs in hdf5
% symbols
fileID = fopen('./data/symbols_training.bin','rb'); % opens with access type read
B = fread(fileID, [N, 0.8*amount], 'double'); % read file
fclose(fileID); % close file
h5create("data/training.h5","/symbols", [N, 0.8*amount]) % create hdf5
h5write("data/training.h5","/symbols", B) % save BBIs in hdf5
% words
fileID = fopen('./data/words_training.bin','rb'); % opens with access type read
C = fread(fileID, [length(words), 0.8*amount], 'double'); % read file
fclose(fileID); % close file
h5create("data/training.h5","/words", [length(words), 0.8*amount]) % create hdf5
h5write("data/training.h5","/words", C) % save BBIs in hdf5
%nlp
for n = 1:length(nlp_name)
    name_nlp_5min = strcat('./data/', nlp_name(n), '_training.bin'); % non-linear parameters
    fileID = fopen(name_nlp_5min,'rb'); % opens with access type read
    D = fread(fileID, [1,0.8*amount], 'double'); % read file
    fclose(fileID); % close file
    h5create("data/training.h5", strcat('/', nlp_name(n)), [1,0.8*amount]) % create hdf5
    h5write("data/training.h5", strcat('/', nlp_name(n)), D) % save ECGs in hdf5
end
h5writeatt("data/training.h5", "/", 'BBI_number_beats', uint32(N))
h5writeatt("data/training.h5", "/", 'Noise', uint32(0))
h5writeatt("data/training.h5", "/", 'samplerate', uint32(sfecg))
h5writeatt("data/training.h5", "/", 'amount', uint32(0.8*amount))
h5writeatt("data/training.h5", "/", 'ecg_duration', uint32(ecg_duration))

% test
% ecg
fileID = fopen('./data/ECG_test.bin','rb'); % opens with access type read
A = fread(fileID, [ecg_duration,0.2*amount], 'double'); % read file
fclose(fileID); % close file
delete data/test.h5
h5create("data/test.h5","/ECG", [ecg_duration,0.2*amount]) % create hdf5
h5write("data/test.h5","/ECG", A) % save ECGs in hdf5
% time
fileID = fopen('./data/t_test.bin','rb'); % opens with access type read
A = fread(fileID, [ecg_duration,0.2*amount], 'double'); % read file
fclose(fileID); % close file
h5create("data/test.h5","/time", [ecg_duration,0.2*amount]) % create hdf5
h5write("data/test.h5","/time", A) % save ECGs in hdf5
% peaks
% classification of waves, peaks and otherwise
fileID = fopen('./data/ipeaks_test.bin','rb'); % opens with access type read
A = fread(fileID, [ecg_duration,0.2*amount], 'int'); % read file
fclose(fileID); % close file
h5create("data/test.h5","/RP", [ecg_duration,0.2*amount]) % create hdf5
h5write("data/test.h5","/RP", A) % save classification in hdf5
%BBI
fileID = fopen('./data/BBI_test.bin','rb'); % opens with access type read
A = fread(fileID, [N, 0.2*amount], 'double'); % read file
fclose(fileID); % close file
h5create("data/test.h5","/BBI", [N, 0.2*amount]) % create hdf5
h5write("data/test.h5","/BBI", A) % save BBIs in hdf5
% symbols
fileID = fopen('./data/symbols_test.bin','rb'); % opens with access type read
B = fread(fileID, [N, 0.2*amount], 'double'); % read file
fclose(fileID); % close file
h5create("data/test.h5","/symbols", [N, 0.2*amount]) % create hdf5
h5write("data/test.h5","/symbols", B) % save BBIs in hdf5
% words
fileID = fopen('./data/words_test.bin','rb'); % opens with access type read
C = fread(fileID, [length(words), 0.2*amount], 'double'); % read file
fclose(fileID); % close file
h5create("data/test.h5","/words", [length(words), 0.2*amount]) % create hdf5
h5write("data/test.h5","/words", C) % save BBIs in hdf5
%nlp
for n = 1:length(nlp_name)
    name_nlp_5min = strcat('./data/', nlp_name(n), '_test.bin'); % non-linear parameters
    fileID = fopen(name_nlp_5min,'rb'); % opens with access type read
    D = fread(fileID, [1,0.2*amount], 'double'); % read file
    fclose(fileID); % close file
    h5create("data/test.h5", strcat('/', nlp_name(n)), [1,0.2*amount]) % create hdf5
    h5write("data/test.h5", strcat('/', nlp_name(n)), D) % save ECGs in hdf5
end
% Write Header in HDF5 Attribute
h5writeatt("data/test.h5", "/", 'BBI_number_beats', uint32(N))
h5writeatt("data/test.h5", "/", 'Noise', uint32(0))
h5writeatt("data/test.h5", "/", 'samplerate', uint32(sfecg))
h5writeatt("data/test.h5", "/", 'amount', uint32(0.2*amount))
h5writeatt("data/test.h5", "/", 'ecg_duration', uint32(ecg_duration))