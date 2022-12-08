function result = pNNlx(hrv_data, threshold)
%pNNlX threshold in ms.

count = 0;
for i=1:length(hrv_data)-1
    if abs(hrv_data(i+1) - hrv_data(i)) < threshold
        count = count + 1;
    end
end
result = count / ( length(hrv_data) -1 ); %Bug für Bug off by one (-1 weglassen für 2.0)
end
