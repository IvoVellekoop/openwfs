function [matrix, matrix_std] = find_mean_shift(frames, opt)
% Calculate transformation matrix
% 
% The matrix elements are computed as the mean of shifts between successive images along the 4th
% dimension of a 4-D array (frames). Furthermore, the standard deviation is given as output for
% each matrix element.
%
% Every shift must be of approximately the same magnitude. If the found shifts differ too much
% (i.e. the tolerance is exceeded), a warning is displayed. This behaviour can be suppressed.
%
% Note: This function is only valid for: number of device axes = frame dimension = 2, i.e. 2D data.
%
% INPUTS
% frames:           A 4-D matrix of numeric elements, containing the recorded frames.
%                   Dimension 1 and 2 are reserved for the frame dimensions. Dimension 3 is for the
%                   device axes (i.e. X & Y dimension of stage or SLM). Dimension 4 is for the
%                   repetitions of the shift measurement.
%
% OPTIONAL INPUTS
% opt.tolerance:    tolerance in deviation (from measurement-to-measurement along each axis)
%                   if sumsqr(S)/sumsqr(M) > tolerance, a warning message may be displayed
% opt.warning:      case-insensitive string anything OTHER THAN "NO" to display a warning
%
% OUTPUTS
% matrix:           Mean displacement as a (2x2) matrix
% matrix_std:       standard deviation of displacement as a (2x2) matrix
%
%
% TEST SCRIPT
% tpm\calibration\tests\test_find_mean_shift.m


arguments
    frames (:,:,:,:) {mustBeNumeric}
    opt.tolerance (1,1) { mustBePositive } = 0.05   % Normalized standard deviation
    opt.warning (1,1) string = "YES"
end

[~, ~, number_of_axes, number_of_frames] = size(frames);
number_of_shifts = number_of_frames-1;

% Initialization
% Note: Only valid for number_of_axes = frame dimension = 2, i.e. 2D data
frame_shifts = zeros(number_of_axes, number_of_axes, number_of_shifts);   % Frame shifts [pixels]

% Loop over frames (device axes and shifts)
for index_axis = 1:number_of_axes
    for index_frame = 1:number_of_shifts
        frame1 = frames(:, :, index_axis, index_frame);       % Current frame
        frame2 = frames(:, :, index_axis, index_frame+1);     % Next frame

        % Compute cross correlation between current & next frame
        cross_correlation = wiener_khinchin(frame1, frame2);

        % Cross-correlation between two similar images will have a peak. 
        % And the offset of the peak will be the shift between the images. (1,1) = no offset
        frame_shifts(:, index_axis, index_frame) = calculate_offset_in_peak(cross_correlation);
    end
end

% Compute matrix from mean shifts
matrix = mean(frame_shifts, length(size(frames))-1);
matrix_std = std(frame_shifts, 1, length(size(frames))-1);

% Check Normalized Root Mean Square of Standard Deviation of matrix elements
error = sqrt(sumsqr(matrix_std) / sumsqr(matrix));
if upper(opt.warning) ~= "NO"
    if  error > opt.tolerance
        warning("Too much deviation between the measurements" + newline + ...
            "Shift between the frames are not repeatable. Sort it out."+ newline +...
            "OR you can increase the sopt.tolerance accordingly (>" + num2str(error) + ")," + newline + ...
            "   Eg: find_mean_shift(frames,'tolerance'," + num2str(error+1) +") "+ newline +...
            "OR set sopt.warning = 'NO' in case you want to neglect the warning" + newline + ...
            "  Eg: find_mean_shift(frames,'warning','no') if you are sure the measurements are correct")
    end
end

end

