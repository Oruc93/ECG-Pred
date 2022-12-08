function [out] = spectral_sum(spectrum, from, to, window_length, interpol_ms)
%SPECTRAL_CUM_SUM Summary of this function goes here
%   Detailed explanation goes here
    out = 0;
    lower_limit = round(from * window_length * (interpol_ms / 1000));
    if lower_limit < 1
        lower_limit = 1;
    end
    upper_limit = round(to * window_length * (interpol_ms / 1000)) - 1;
    "lower limit: " + lower_limit
    "upper limit: " + upper_limit
    for t = lower_limit + 1 : upper_limit + 1
        out = out + spectrum(t);
    end
    
end

