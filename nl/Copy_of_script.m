close all;
clearvars;

res = [];
for i = 1:35
    if i < 10
        string = sprintf('..\\data\\noNNtime_zero\\00%d.hrv', i);
    
    else
        string = sprintf('..\\data\\noNNtime_zero\\0%d.hrv', i);
    end
    fileID = fopen(string, 'r');
    formatSpec = '%f';
    data = fscanf(fileID, formatSpec);
    
    [words, symbols] = calc_symboldynamics(data, 0.01, "movdiff");
    forbword = forbidden_words(words);
    fwshannon_param = fwshannon(words);
    fwrenyi_025_param = fwrenyi(words, 0.25);
    fwrenyi_4_param = fwrenyi(words, 4);
    wpsum02_param = wpsum02(words);
    wpsum13_param = wpsum13(words);
    wsdvar_param = wsdvar(symbols);
    plvar_5_param = plvar(data, 5);
    plvar_10_param = plvar(data, 10);
    plvar_20_param = plvar(data, 20);
    phvar_20_param = phvar(data, 20);
    phvar_50_param = phvar(data, 50);
    phvar_100_param = phvar(data, 100);
    
    row = [
        forbword, fwshannon_param, fwrenyi_025_param, fwrenyi_4_param, wsdvar_param, wpsum02_param, wpsum13_param, plvar_5_param, plvar_10_param, plvar_20_param, phvar_20_param, phvar_50_param, phvar_100_param        
    ];
    res = [res; row];
end

res = array2table(res, "VariableNames", { 
    'forbword', 'fwshannon', 'fwrenyi 0.25', 'fwrenyi 4', 'wsdvar', 'wpsum 02', 'wpsum 13', 'plvar 5', 'plvar 10', 'plvar 20', 'phvar 20', 'phvar 50', 'phvar 100'
    });

writetable(res, "..\\results\\nl.xls", "sheet", 1);