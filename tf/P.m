function [out] = P(spectrum, window_length, interpol_ms)
%Powersum spectrum from 0.04 to 0.15 HZ
%   Detailed explanation goes here
out = spectral_sum(spectrum, 0.00, 0.40, window_length, interpol_ms);
end

