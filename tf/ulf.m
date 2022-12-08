function [out] = ulf(spectrum, window_length, interpol_ms)
%ULF Ultra low frequency from 0 HZ to 0.0033 HZ
%   Detailed explanation goes here
out =  spectral_sum(spectrum, 0, 0.0033, window_length, interpol_ms);

