function [out] = lf(spectrum, window_length, interpol_ms)
%VLF low frequency spectrum from 0.04 to 0.15 HZ
%   Detailed explanation goes here
out = spectral_sum(spectrum, 0.04, 0.15, window_length, interpol_ms);
end



