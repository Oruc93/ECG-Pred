function [output] = fft_averaging(data, window_length, window_shift, number_of_avg)
%FFT_AVERAGING Summary of this function goes here
%   Detailed explanation goes here

exponent_of_2 = 2;
window_sampling_index = 4;

%finding the power of 2
while window_sampling_index - window_length < 0
    exponent_of_2 = exponent_of_2 + 1;
    window_sampling_index = window_sampling_index * 2;
end



fft_total = zeros(window_length, 1);
for i = 1:number_of_avg
    window_sampling_index = (i-1)*window_shift + 1; %Starting point of current window shift

    current_window = data(window_sampling_index:window_sampling_index + window_length - 1);
    blackman_harris = window_function(window_length);
    blackman_harris = blackman_harris(512:1023);
    current_window = current_window .* blackman_harris;
    
    fft_of_window = fft(current_window);
    
    fft_total = fft_total + abs(fft_of_window).^2;
%    for j = 1:length(fft_of_window)
%       
%        fft_total(j) = fft_total(j) + real(fft_of_window(j))^2 + imag(fft_of_window(j))^2;
%    end
end

s = number_of_avg*window_length / 2;
s = s*s;

fft_total = fft_total ./ s;

output = fft_total(1:window_length / 2);






