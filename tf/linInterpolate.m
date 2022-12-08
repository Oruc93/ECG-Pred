function [out] = linInterpolate(x, dx, x1, y1, y2)
%INTERPOLATE Summary of this function goes here
%   Detailed explanation goes here
    if(dx == 0)
        out = 0;
        return;
    end
    p = (x-x1) / dx;
    out = y1 + (y2-y1)*p;
end

