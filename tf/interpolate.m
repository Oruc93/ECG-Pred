function [out] = interpolate(data)
%INTERPOLATE Summary of this function goes here
%   Detailed explanation goes here
interpolate_ms = 500;
abs = 0;
sumdt = 0;
abspos0 = 0;
i = 2;
out = [];

for t=1:length(data)-3
    abspos0 = abspos0 + data(t);
end

aa = data(i);
aaa = data(i);
bb = abs - data(i);
cc =  data(i-1);

while sumdt<=abspos0 && i<length(data)-4
    while abs<sumdt
        abs = abs + data(i);
        aa = data(i);
        aaa = data(i);
        bb = abs - data(i);
        cc = data(i-1);
        i = i + 1;
    end
    out(end+1) = linInterpolate(sumdt, aa, bb, cc, aaa);
    sumdt = sumdt + interpolate_ms;

    
end
out = out.';

end

