function [out] = vlf(spectrum, window_length, interpol_ms)
%VLF very low frequency spectrum from 0.0033 to 0.04HZ
%   Detailed explanation goes here
out = spectral_sum(spectrum, 0.0033, 0.04, window_length, interpol_ms);
end

