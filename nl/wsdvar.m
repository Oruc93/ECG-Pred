function [param] = wsdvar(symbols)
%WSDVAR Summary of this function goes here
%   Detailed explanation goes here
output = [];
%if mod(length(symbols), 3) == 1
%    symbols(end+1) = -1;
%    symbols(end+1) = -1;
%elseif mod(length(symbols, 3)) == 2
%    symbols(end+1) = -1;
%end
%words = reshape(symbols,3, []);
symbols = reshape(symbols, 1, []);
words = vertcat(symbols(1:end-2), symbols(2:end-1), symbols(3:end));

for i=2:length(words)-2 %Bug 4 Bug vs 2.0: Grenzen auf alle wörter ändern
    if ~isempty(find(words(:,i) == -1, 1))
        continue;
    end
    n1 = sum(words(:,i) == 1);
    n3 = sum(words(:,i) == 3);
    first_1 = find(words(:,i) == 1, 1, 'first');
    if isempty(first_1)
        first_1 = 4;
    end
    first_3 = find(words(:,i) == 3, 1, 'first');
    if isempty(first_3)
        first_3 = 4;
    end
    if (n1+n3 == 0) 
        output(end+1) = 0;
    elseif (n1 + n3 == 3 && first_3 > first_1) 
        output(end+1) = 3;
    elseif (n1 + n3 == 2) && (first_3 > first_1)
        output(end+1) = 2;
    elseif (n1 + n3 == 1 && first_3 > first_1)
        output(end+1) = 1;
    elseif ((n1 + n3 == 1) && (first_3 < first_1))
        output(end+1) = -1;
    elseif (n1 + n3 == 2 && first_3 < first_1)
        output(end+1) = -2;
    elseif (n1 + n3 == 3 && first_3 < first_1)
        output(end+1) = -3;                         % Änderung: 3 --> -3
    end
end
param = std(output);
end

