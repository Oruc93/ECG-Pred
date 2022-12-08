function [res] = noNNtime(raw_series, removed_idx)
%NONNTIME Summary of this function goes here
%   Detailed explanation goes here

    sum = 0;
    for ii=1:length(removed_idx)
        sum = sum + raw_series(ii);
    end
    res = sum;
end

