function [forb_words] = forbidden_words(words)
%FORBIDDEN_WORDS Summary of this function goes here
%   Detailed explanation goes here
forb_words = 0;
word_sum = sum(words);
for i=1:length(words)
    if words(i) / word_sum < 0.001
        forb_words = forb_words + 1;
    end
end
end

