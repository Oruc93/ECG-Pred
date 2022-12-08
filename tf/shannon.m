function result = shannon(hrv_data, hist_length)
%SHANNON Summary of this function goes here
%   Detailed explanation goes here

%no_of_hist_values = (max(hrv_data) - min(hrv_data)) / hist_length;
%edges = linspace(0, 2000, 101);
[~, edges] = unique(round((1:2000) .* (99/2000)));
[N,~] = histcounts(hrv_data, edges, "Normalization", "probability");

%total = sum(N);
shannonValue = 0;
for ii=1:length(N)
    if N(ii) ~= 0
        shannonValue = shannonValue - N(ii) * log(N(ii));
        %shannonValue = shannonValue + (N(ii) / total * log(N(ii) / total));
    end
end
result = shannonValue;

end

