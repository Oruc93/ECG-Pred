function [output] = pNNx(hrv_data, threshold)
%PNNX threshold in ms.

count = 0;
for i=1:length(hrv_data)-1
    if abs(hrv_data(i+1) - hrv_data(i)) > threshold
        count = count + 1;
    end
end
output = count / length(hrv_data);
end

