% close all;
% clearvars;


res = [];
i=21;
%for i = 1:35
    if i < 10
        string = sprintf('..\\data\\noNNtime_zero\\00%d.hrv', i);
    
    else
        string = sprintf('..\\data\\noNNtime_zero\\0%d.hrv', i);
    end
    fileID = fopen(string, 'r');
    formatSpec = '%f';
    data = fscanf(fileID, formatSpec);


    %data = data ./ 1000;

    interpolated = interpolate(data);

    spec_window_length = 512;
    no_of_averaging = 6;

    %mean subtraction
    sing = sum(interpolated) / (length(interpolated)-1);                            %Änderung length(interpolated) --> length(interpolated)-1
    interpolated = interpolated - sing;

    window = window_function(spec_window_length);
    window = window(512:1023);
    %plot(window);
    %figure();



    %[fft_data, freq] = pwelch(interpolated, window, [], [], 2);

    window_shift = min( fix((length(interpolated)-1 - 1 - spec_window_length - 1) / 5), spec_window_length);    %Änderung length(interpolated) -->length(interpolated) -1
    fft_data = fft_averaging(interpolated, spec_window_length, window_shift, 6);
    %S_matlab = 1.5692e+04;
    %S_pascal = 7.512908;
    %fft_data = fft_data * S_pascal / S_matlab;

    %plot(fft_data)

    interpol_ms = 500;

    P_param = P(fft_data, spec_window_length, interpol_ms);

    ulf_param = ulf(fft_data, spec_window_length, interpol_ms);
    vlf_param = vlf(fft_data, spec_window_length, interpol_ms);
    lf_param = lf(fft_data, spec_window_length, interpol_ms);
    hf_param = hf(fft_data, spec_window_length, interpol_ms);

    ulf_P_param = ulf_param / P_param;
    vlf_P_param = vlf_param / P_param;
    lf_P_param = lf_param / P_param;
    hf_P_param = hf_param / P_param;
    
    row = [
        ulf_param, vlf_param, lf_param, hf_param, ulf_P_param, vlf_P_param, lf_P_param, hf_P_param, P_param
        ];
    res = [res; row];

% andere Parameter

%     lf_hf_param = lf_param / hf_param;
%     ulf_lf_hlf_P_param = ( ulf_param + vlf_param + lf_param ) / P_param;
%     ulf_vlf_P_param = ( ulf_param + vlf_param ) / P_param;
%     uvlf_param = ulf_param + vlf_param;
%     lfn_param = lf_param / ( lf_param + hf_param);
%     hfn_param = hf_param / ( lf_param + hf_param);

%end

res = array2table(res, 'VariableNames', {
    'ULF', 'VLF', 'LF', 'HF', 'ULF/P', 'VLF/P', 'LF/P', 'HF/P', 'P'});

%writetable(res, "..\\results\\tf.xls", "sheet", 1);


