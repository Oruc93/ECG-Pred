function [output] = cvNN(hrv_data)
    output = sdNN(hrv_data) / meanNN(hrv_data);
end

