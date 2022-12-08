res = [];
for i=1:35
    if i < 10
        string = sprintf('..\\data\\noNNtime_zero\\00%d.hrv', i);
    
    else
        string = sprintf('..\\data\\noNNtime_zero\\0%d.hrv', i);
    end
    fileID = fopen(string, 'r');
    
    formatSpec = '%f';
    raw = fscanf(fileID, formatSpec);
    A = raw;
    %[A, rmvd_idx] = adaptive_hrv_filter(raw);
    %rmvd_idx
    
    %Save adaptive filtered series
    if i < 10
        out = sprintf('..\\results\\filtered\\00%d.adp', i);
    
    else
        out = sprintf('..\\results\\filtered\\0%d.adp', i);
    end
    [fid, msg] = fopen(out, 'wt');
    fprintf(fid, '%d\n', A);
    fclose(fid);
    
    if i == 1
        figure()
        plot(A)
    end
    
    row = [
        %noNNtime(raw, rmvd_idx) 
        meanNN(A) sdNN(A) cvNN(A) sdaNNx(A, 1) sdaNNx(A, 5) sdaNNx(A, 10) rmssd(A) pNNx(A, 50) pNNx(A, 100) pNNx(A, 200) shannon(A, 1) renyix(A, 0.25) renyix(A, 4) renyix(A, 2) pNNlx(A, 10) pNNlx(A, 20) pNNlx(A, 30) pNNlx(A, 50)];
    res = [res; row];
    
end
res = array2table(res,'VariableNames',{
    %'noNNtime',
    'meanNN','sdNN', 'cvNN', 'sdaNN1', 'sdaNN5', 'sdaNN10', 'rmssd', 'pNN50', 'pNN100', 'pNN200', 'shannon', 'renyix 0.25', 'renyix 4', 'renyix2', 'pNNl10', 'pNNl20', 'pNNl30', 'pNNl50' });
writetable(res, 'results\\results.xls', 'sheet', 2);
