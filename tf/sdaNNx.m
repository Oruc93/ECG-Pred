function [output] = sdaNNx(hrv_data, window_length)
%Window length in minutes.

%sum(hrv_data);
start_index = 1;
time_sum = 0;
mean_values = [];
for j = 1:length(hrv_data)-1                                        %Änderung: j_max->j_max-1
    time_sum = time_sum + hrv_data(j);
    if time_sum > window_length * 60 * 1000
        mean_values(end+1) = mean(hrv_data(start_index:j+1));       %Änderung: j->j+1
        start_index = j+1; %Bug für bug: j, 2.0: j+1                %Änderung: j->j+1
        time_sum = 0;          
    end
end
if start_index < length(hrv_data) %Evtl Bug für Bug: Soll das letzte, angebrochene Interval drin sein?
    mean_values(end+1) = mean(hrv_data(start_index:end));
end
if length(mean_values) > 1
    output = std(mean_values(1:end)); 
else
    output = 0;
end
end

