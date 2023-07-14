function rawdata = calibrate_matrices_acquisition(slm, stage, hSICtl, hSI, opt)
% Inputs:
%   slm                     SLM object (hardware repo).
%   hSICtl                  ScanImage Control handle.
%   hSI                     ScanImage handle.
%   options:
%     repetitions             Number of repetitions of each shifting experiment
%     pixels_per_period_slm   Pixels per period of the SLM gradient.


arguments
    slm (1,1)    SLM
    opt.repetitions (1,1) { mustBePositive } = 5    % Number stage shifts to be recorded per axis (average will be computed)
    opt.delta_slope (1,1) = 255/50;                 % slope of the gradient pattern [2*pi/SLM pixels]
end

%% add relevant paths
addpath('C:\git\tpm\calibration\functions\');


%% Construct SLM gradient pattern
bg_patch_id = 1;                            % Gradient Patch ID
rawdata.slm_height_pix = size(slm.getPixels, 2);    % SLM height in pixels
rawdata.slope = [-2,-1,0,1,2]'*opt.delta_slope;
slm_pattern_gradient = rawdata.slope.*(0:rawdata.slm_height_pix);

%% Initializations
rawdata.frames_gradient = zeros(rawdata.frame_width_pix, rawdata.frame_width_pix, 2, 5);

%% acquire gradient images
% set square patch normalized to the height of the slm
slm.setRect(bg_patch_id,[0,0,1,1])

% loop over horizontal and vertical 
for index_gradient_axis = 1:2

    for index_frame = 1:5
        if index_gradient_axis == 1
            slm.setData(bg_patch_id, slm_pattern_gradient(index_frame,:)); slm.update;
            rawdata.frames_gradient(:,:,index_gradient_axis,index_frame) = grabSIFrame(hSI,hSICtl);
        else
            slm.setData(bg_patch_id, slm_pattern_gradient(index_frame,:)'); slm.update;
            rawdata.frames_gradient(:,:,index_gradient_axis,index_frame) = grabSIFrame(hSI,hSICtl);
        end
    end
end

%% Creating a flat SLM pattern again
slm.setData(bg_patch_id, 0); slm.update;