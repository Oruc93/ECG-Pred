function [shannon] = fwshannon(words)
%FWSHANNON Summary of this function goes here
%   Detailed explanation goes here
shannon = 0;
wordsum = sum(words);
for i=1:length(words)
    if words(i) > 0
        shannon = shannon - words(i) / wordsum * log(words(i) / wordsum);
    end
end
end

