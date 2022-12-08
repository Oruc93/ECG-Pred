function [output] = rmssd(hrv_data)

differences = [];
for i=1:length(hrv_data)-1
    differences = [differences; hrv_data(i+1) - hrv_data(i)];
end
differences = differences.^2;
output = sqrt(sum(differences) / length(differences));
end

