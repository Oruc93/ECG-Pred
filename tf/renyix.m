function result = renyix(hrv_data, order)
%SHANNON Summary of this function goes here
%   Detailed explanation goes here

%no_of_hist_values = (max(hrv_data) - min(hrv_data)) / hist_length;
[~, edges] = unique(round((1:2000) .* (99/2000)));
%edges = linspace(0, 2000, 101);
[N,~] = histcounts(hrv_data, edges, "Normalization", "probability");
renyixValue = 0;
for ii=1:length(N)
    if N(ii) ~= 0
        renyixValue = renyixValue + N(ii)^order;
    end
end

result = 1 / (1-order) * log(renyixValue);

end
