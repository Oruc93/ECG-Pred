function [renyi] = fwrenyi(words, alpha)
%FWRENYI Summary of this function goes here
%   Detailed explanation goes here
if alpha>0 && alpha ~= 1
    wordsum = sum(words);
    renyi = 0;
    for i=1:length(words)
        if words(i) > 0
            renyi = renyi + exp(log(words(i) / wordsum) * alpha);
        end
    end
    renyi = (1/(1-alpha)) * log(renyi);
end
end

