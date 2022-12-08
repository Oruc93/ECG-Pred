function [param] = wpsum13(words)
%WPSUM Summary of this function goes here
%   Detailed explanation goes here

param = 0;
arrpos = 1;
for i1=0:3
    for i2=0:3
        for i3 = 0:3
            if ((i1 == 1) || (i1==3)) && ((i2 == 1) || (i2 == 3)) && ((i3 == 1) || (i3 == 3))
                param = param + words(arrpos);
                
            end
            arrpos = arrpos + 1;
        end
    end
end
param = param / sum(words);

