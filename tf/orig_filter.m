function [varargout] = adaptive_hrv_filter(hrv_timeseries, varargin)

%% Configuration
    %Default Values
    CONFIG = struct();
    CONFIG.replace_nonnormal_values=true;
    CONFIG.adaptivity_controlling_coefficient = 0.05;
    CONFIG.range_proportionality_limit=10/100;
    CONFIG.outlier_min_z_factor=3;
    CONFIG.allowed_excess_hrv_variability=20; %ms
    CONFIG.remove_nonphysiological_outliers=true; %Helps with stability
    CONFIG.mimimum_physiological_value=200; %ms
    CONFIG.maximum_physiological_value=2000; %ms
    
    %Parse Configuration parameters
    for ii = 1:2:numel(varargin)
        if isempty(varargin{ii})
            continue;
        end    
        switch lower(varargin{ii})
            case 'replace_nonnormal_values'
                    assert(islogical(varargin{ii+1}), 'adaptive_hrv_filter:unrecognized_config', 'adaptive_hrv_filter: parameter \"replace_nonnormal_values\" requires a logical value');
                    CONFIG.replace_nonnormal_values=varargin{ii+1};
            case 'remove_nonphysiological_outliers'
                    assert(islogical(varargin{ii+1}), 'adaptive_hrv_filter:unrecognized_config', 'adaptive_hrv_filter: parameter \"remove_nonphysiological_outliers\" requires a logical value');
                    CONFIG.remove_nonphysiological_outliers=varargin{ii+1};
            otherwise
                error('adaptive_hrv_filter:unrecognized_config', 'adaptive_hrv_filter: Unrecognized parameter-value pair specified.')
        end
    end

%% First Parameter Check
    if ~isvector(hrv_timeseries) || ~isnumeric(hrv_timeseries)
        error('cvp_utils:normalize_hrv_timeseries:hrv_not_vector', 'hrv_timeseries is not a numeric vector');
    end
    
    if (~sum(size(hrv_timeseries) > 1) == 1)
        error('cvp_utils:normalize_hrv_timeseries:hrv_not_vector', 'hrv_timeseries is not a numeric vector with a single non-singular dimension.');
    end
    
    hrv_timeseries = reshape(double(hrv_timeseries), [], 1);    
    
    
%% Parameter Checks (just to be on the safe side)
    assert(length(size(hrv_timeseries)) == 2 && size(hrv_timeseries, 2) == 1);


%% Remove values that are out of range of Physiological Heart beats
    removed_nonphysiological_outliers = false(size(hrv_timeseries));
    removed_nonphysiological_outliers_idx = [];
    if CONFIG.remove_nonphysiological_outliers
        removed_nonphysiological_outliers = (hrv_timeseries > CONFIG.maximum_physiological_value | hrv_timeseries < CONFIG.mimimum_physiological_value);
        removed_nonphysiological_outliers_idx = find(removed_nonphysiological_outliers);
        hrv_timeseries(hrv_timeseries > CONFIG.maximum_physiological_value | hrv_timeseries < CONFIG.mimimum_physiological_value) = [];
    end
    
%% Calculate helper matrixes for the Binominal Filter
    bin_coeff = [1; 6; 15; 20; 15; 6; 1];
    coeff_count=length(bin_coeff);
    coeff_sum=sum(bin_coeff);
    assert(mod(coeff_count, 2) == 1);
    timeseries_length=length(hrv_timeseries);

    hrv_filter_value_mat = repmat(bin_coeff, 1,timeseries_length);
    
    value_index_column = reshape((1:coeff_count)-ceil(coeff_count/2), [], 1);
    hrv_ind_offset_val = repmat(value_index_column, 1,timeseries_length);
    hrv_ind_base_val = repmat(1:timeseries_length, coeff_count, 1);

    %The first and last elements will be out of range...
    %...work around by just (re-)using first and last elements...
    hrv_idx_mat = hrv_ind_offset_val + hrv_ind_base_val;
    hrv_idx_mat(hrv_idx_mat < 1) = 1;
    hrv_idx_mat(hrv_idx_mat > timeseries_length) = timeseries_length;

%% Calculate first filtered Signal (Through Binominal Filter)
    filtered_hrv=reshape(sum(hrv_timeseries(hrv_idx_mat) .* hrv_filter_value_mat, 1) .* (1/coeff_sum), [], 1);

    
%% Apply "Adaptive Percent Filter" => fixed_hrv_timeseries
    [adaptive_mean, adaptive_sigma] =  calculate_adaptive_moments(filtered_hrv, coeff_count, CONFIG);
    adaptive_sigma_mean = mean(adaptive_sigma);

    last_good_value=hrv_timeseries(1); %Maybe filtered_hrv(1)
    last_good_range=...
            CONFIG.range_proportionality_limit * last_good_value + ...
            CONFIG.outlier_min_z_factor * adaptive_sigma_mean;    
    
    hrv_diff=diff(hrv_timeseries);
    normal_hrv_values=true(size(filtered_hrv));
    for ii=2:length(hrv_timeseries)
        current_diff=abs(hrv_diff(ii-1));
        current_max_range=...
            CONFIG.range_proportionality_limit * hrv_timeseries(ii-1) + ...
            CONFIG.outlier_min_z_factor * adaptive_sigma_mean;
        
        current_value_is_normal= current_diff <= current_max_range || current_diff <= last_good_range;
        if (current_value_is_normal)
            last_good_value=hrv_timeseries(ii);
            last_good_range=...
                    CONFIG.range_proportionality_limit * last_good_value + ...
                    CONFIG.outlier_min_z_factor * adaptive_sigma_mean; 
        end
        normal_hrv_values(ii) = current_value_is_normal;
    end
 
    fixed_hrv_timeseries = hrv_timeseries;
    non_normal_hrv_idxs = reshape(find(~normal_hrv_values), [], 1);
    fixed_hrv_timeseries(non_normal_hrv_idxs) = adaptive_mean(non_normal_hrv_idxs) + (rand(size(non_normal_hrv_idxs)) - 0.5) .* adaptive_sigma(non_normal_hrv_idxs);

    
%% Calculate second filtered Signal (Through Binominal Filter)
    filtered_fixed_timeseries=reshape(sum(fixed_hrv_timeseries(hrv_idx_mat) .* hrv_filter_value_mat, 1) .* (1/coeff_sum), [], 1);

    
%% Apply "Adaptive Controlling Procedure" => fixed_fixed_hrv_timeseries
    [adaptive_fixed_mean, adaptive_fixed_sigma] =  calculate_adaptive_moments(filtered_fixed_timeseries, coeff_count, CONFIG);
    
    normal_fixed_hrv_values = ...
        abs(fixed_hrv_timeseries-adaptive_fixed_mean) <= ...
        (CONFIG.outlier_min_z_factor .* adaptive_fixed_sigma + ...
        CONFIG.allowed_excess_hrv_variability);
        
    fixed_fixed_hrv_timeseries = fixed_hrv_timeseries;
    non_normal_fixed_hrv_idxs = reshape(find(~normal_fixed_hrv_values), [], 1);
    fixed_fixed_hrv_timeseries(non_normal_fixed_hrv_idxs) = ...
        adaptive_fixed_mean(non_normal_fixed_hrv_idxs) + ...
        (rand(size(non_normal_fixed_hrv_idxs)) - 0.5) .* adaptive_fixed_sigma(non_normal_fixed_hrv_idxs);

    
%% Set Return Values

    

    non_normal_hrv_idxs=find((~normal_fixed_hrv_values) | (~normal_hrv_values));

    if (CONFIG.replace_nonnormal_values)
        %Return a fixed time series, a list of indices (relative to the returned list) that were changed to
        %create such time series, and a list of indices that were removed
        %from the input time series, because they were so out of range that
        %they would mess with the filters.
        
        varargout{1} = fixed_fixed_hrv_timeseries;
        if (nargout > 1)
            varargout{2} = non_normal_hrv_idxs;
        end
        if (nargout > 2)
            varargout{3} = removed_nonphysiological_outliers_idx;
        end
    else
        %Return a list of indices relative to the input data, of data
        %points that were either removed, or changed toward normality.
        
        non_normal_input_data = true(size(removed_nonphysiological_outliers));
        non_removed_data_points = setdiff(1:length(non_normal_input_data), removed_nonphysiological_outliers_idx);
        
        non_removed_non_normal_input_data = false(size(fixed_fixed_hrv_timeseries));
        non_removed_non_normal_input_data(non_normal_hrv_idxs) = true;
        
        non_normal_input_data(non_removed_data_points) = non_removed_non_normal_input_data;
        
        varargout{1} = find(non_normal_input_data);
    end
    
end            

%% Helper Function to Calculate Adaptive Moments
function [adaptive_mean, adaptive_sigma] = calculate_adaptive_moments(timeseries, initialization_length, CONFIG)
    adaptive_mean=nan(size(timeseries));
    adaptive_variance=nan(size(timeseries));
    %Initialize...
    adaptive_initialization_idxs=1:initialization_length;
    adaptive_mean(1) = mean(timeseries(adaptive_initialization_idxs));
    adaptive_variance(1) = mean((adaptive_mean(1) - timeseries(adaptive_initialization_idxs)) .^ 2);

    for jj=2:length(timeseries)
        adaptive_mean(jj) = adaptive_mean(jj-1) - ...
            CONFIG.adaptivity_controlling_coefficient * (adaptive_mean(jj-1) - timeseries(jj-1));
        
        last_variance_item=(adaptive_mean(jj-1) - timeseries(jj-1)) ^ 2;
        adaptive_variance(jj) = adaptive_variance(jj-1) - ...
            CONFIG.adaptivity_controlling_coefficient * (adaptive_variance(jj-1) - last_variance_item);
    end  
    adaptive_mean=reshape(adaptive_mean, [], 1);
    adaptive_sigma = reshape(sqrt(adaptive_variance), [], 1);
%   figure(); subplot(2,1,1); plot(adaptive_sigma, 'g'); subplot(2,1,2); plot(adaptive_mean, 'g'); hold on; plot(timeseries, 'k')
end


% Author: Jan F. Kraemer <jan.kraemer@physik.hu-berlin.de>
% Based on: Wessel, N., Voss, A., Malberg, H., Ziehmann, Ch., 
%           Voss, H. U., Schirdewan, A., Meyerfeldt, U.,
%           Kurths, J.: 
%           Nonlinear analysis of complex phenomena in cardiological data, 
%           Herzschr. Elektrophys., 11(3), 2000, 159-173, 
%           doi:10.1007/s003990070035.