% Code to test the function find_mean_shift()
% The function is used to calculate the mean of 
% shifts between successive images in a multidimensional array
% tpm\calibration\functions\find_mean_shift.m


%%
clc; clear; close all

%%
addpath(fileparts(fileparts(mfilename('fullpath'))) + "/functions")

%%  Gaussian
x = -0.5:0.005:0.51;
y = -0.5:0.005:0.51;
noiselevel = 0.01;
test_Gaussian = exp(-5 * (x'.^2 + y.^2));

number_of_axes = 2; 
number_of_repetitions = 9;

shift_along_axis{1} = [10 1];
shift_along_axis{2} = [-5 4];

M_given = [shift_along_axis{1}' shift_along_axis{2}'];

[sx, sy] = size(test_Gaussian);

%% Different test cases of shifts
disp('=== TEST 1: correct shifts ===')
frames = zeros(sx, sy ,number_of_axes,number_of_repetitions);

for count_axis = 1:number_of_axes
    shift = shift_along_axis{count_axis};
    for count_repetitions = 1:number_of_repetitions+1
        frames(:,:,count_axis,count_repetitions) = ...
            circshift(test_Gaussian, count_repetitions * shift)...  % Shift test image
            + randn(length(x)) * noiselevel;                        % Add some white noise
    end
end

[M_calculated, mat_std] = find_mean_shift(frames);

% M = 5 % " To test the test"
if(M_calculated == M_given)
    disp("Given shifts = ")
    disp(M_given)
    disp("Calculated shifts = ")
    disp(M_calculated)
    disp("Test passed :)")
else
    disp("Given shifts = ")
    disp(M_given)
    disp("Calculated shifts = ")
    disp(M_calculated)
    error('!!!!! Test failed !!!')
end


%% Test incorrect shift
fprintf('\n=== TEST 2: tolerance ===\n')
[sx, sy] =size(test_Gaussian);
frames_wrong = zeros(sx, sy ,number_of_axes,number_of_repetitions);
random_shift = 10;      % Maximum random shift to give each frame. Set to 1 to make test fail

for count_axis = 1:number_of_axes
    shift = shift_along_axis{count_axis};
    for count_repetitions = 1:number_of_repetitions+1
        frames_wrong(:,:,count_axis,count_repetitions) = ...
            circshift(test_Gaussian, randi(random_shift) * count_repetitions * shift)... % Shift test image
             + randn(length(x)) * noiselevel;           % Add some whitenoise
    end
end

lastwarn('')         % Reset last warning
disp('Note: This test should trigger a warning message.')
[M_calculated_wrong, mat_std_wrong] = find_mean_shift(frames_wrong);


% M = 5 % " To test the test"
if strfind(lastwarn, 'Too much deviation between the measurements')
    disp("Given shifts = ")
    disp(M_given);
    disp("Calculated wrong shifts = ")
    disp(M_calculated_wrong)
    disp("Test passed :)")
else
    disp("Given shifts = ")
    disp(M_given);
    disp("Calculated wrong shifts = ")
    disp(M_calculated_wrong)
    error('!!!!! Test failed !!!')
end
