function [param] = phvar(hrv_data, threshold)
%PHVAR Summary of this function goes here
%   Detailed explanation goes here
symbols = zeros(1, length(hrv_data) - 1);
for i=1:length(hrv_data)-1
    if abs(hrv_data(i) - hrv_data(i+1)) <= threshold
        symbols(i) = 1;
    end
end

%param = 0;
params = zeros(1, length(hrv_data) - 7);
for i=1:length(params)
    si = sum(symbols(i:i+5));
    if si == 0
        params(i) = 1;
    end
end
param = mean(params);
%param = param / (length(symbols) - 6);

end