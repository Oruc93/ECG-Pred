function [w] = window_function(window_length)
%WINDOW_FUNCTION Summary of this function goes here
%   Detailed explanation goes here
N = window_length; L = 3; No2 = (N-1)/2; n=-No2:No2;
ws = zeros(L,3*N); z = zeros(1,N);
for l=0:L-1
  ws(l+1,:) = [z,cos(l*2*pi*n/N),z];
end
alpha = [0.44959,0.49364,0.05677]; % Classic Blackman
w = alpha * ws;
w = w';
end

