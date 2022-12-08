function [param] = wpsum02(words)
%WPSUM Summary of this function goes here
%   Detailed explanation goes here

param = 0;
arrpos = 1;
for i1=0:3
    for i2=0:3
        for i3 = 0:3
            if ((i1 == 0) || (i1==2)) && ((i2 == 0) || (i2 == 2)) && ((i3 == 0) || (i3 == 2))
                param = param + words(arrpos);
                
            end
            arrpos = arrpos + 1;
        end
    end
end
param = param / sum(words);