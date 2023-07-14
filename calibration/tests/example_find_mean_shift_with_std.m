% Code to test the function find_mean_shift()
% C:\Git\tpm\calibration\functions\find_mean_shift.m
% specifically, the part that calculates the standard deviation



%%
clc
clear all
close all

%%
addpath('C:\Git\tpm\calibration\functions')

%%  Gaussian
x = -0.5:0.01:0.51;
y = -0.5:0.01:0.51;
test_Gaussian= exp(-(x'.^2+y.^2));

number_of_axes = 2; 
number_of_repetitions = 7;

shift_along_axis{1} = [40 1];
shift_along_axis{2} = [2 30];

M_given = [shift_along_axis{1}' shift_along_axis{2}'];

% Test fails for large (~1) noise in the shifts
noise_in_shift = 0.1;


%% Different test cases of shifts
[sx, sy] =size(test_Gaussian);
frames = zeros(sx, sy ,number_of_axes,number_of_repetitions);

for count_axis = 1:number_of_axes
    shift = shift_along_axis{count_axis};
    for count_repetitions = 1:number_of_repetitions+1
        linear_shift = count_repetitions*shift;
        random_shift = noise_in_shift*randn(1,2).*shift
        frames(:,:,count_axis,count_repetitions) = circshift(test_Gaussian,floor(linear_shift+random_shift));
    end
end

[M_calculated, S] = find_mean_shift(frames);


