% Code to test the calculation of shift using
% wiener_khinchin.m and calculate_offset_in_peak
% C:\Git\tpm\calibration\functions\wiener_khinchin.m
% C:\Git\tpm\calibration\functions\calculate_offset_in_peak.m

% The default conventions of MATLAB are followed
% (row as the first dimesion and column as the second)


%%
clc
clear all
close all

%%
addpath('C:\git\openwfs\calibration\functions');

%%  Gaussian
x = -0.5:0.01:0.51;
y = -0.5:0.01:0.51;
test_Gaussian= exp(-(x'.^2+y.^2));

%% Different test cases of shifts
test_cases = {[0 0], % No shifts
    [1 3], % odd shifts
    [2 4] , % even shifts
    [-8 -16], % Negative even shifts
    [-5 -9], % Negative odd shifts
    floor(size(test_Gaussian)/2), % edge case
    }


% Not valid as it is > end/2 
% Can be used to "test the test" 
% test_cases{1} = ceil(size(test_Gaussian)/2) ; % edge case
% test_cases{2} = ceil(size(test_Gaussian)/2) + [2,2]

%%

for count_cases = 1:length(test_cases)
    shifts_given = test_cases{count_cases};
    test_Gaussian_shifted = circshift(test_Gaussian,shifts_given);

    correlation = wiener_khinchin(test_Gaussian,test_Gaussian_shifted);

    shifts_calculated = calculate_offset_in_peak(correlation) ;

    flag = 0;
    if(shifts_given == shifts_calculated)
        display("Test successful for " +...
            " Expected shift: " + num2str(shifts_given)  +...
            " Calculated shift: " + num2str(shifts_calculated))
    else
        flag = 1
        display("Test Failed for." + newline + ...
            " Expected shift: " + num2str(shifts_given) + newline + ...
            " Calculated shift: " + num2str(shifts_calculated) + newline + ...
            "_________________________")
    end
end

if (flag==1)
    error('!!!!! Test failed !!!')
end

