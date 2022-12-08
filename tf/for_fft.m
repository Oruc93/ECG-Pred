function [out] = for_fft(data, window_length)

H = blackmanharris(window_length);
%window = generate(data);
if length(data) < window_length
    data(end: window_length) = 0;
end
out = fft(H .* data, window_length);
end

