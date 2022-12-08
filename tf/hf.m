function [out] = hf(spectrum, window_length, interpol_ms)
%HF high frequency spectrum from 0.15 to 0.4 HZ
%   Detailed explanation goes here
out = spectral_sum(spectrum, 0.15, 0.4, window_length, interpol_ms);
end



