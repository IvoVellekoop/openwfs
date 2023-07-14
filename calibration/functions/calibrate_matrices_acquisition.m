function rawdata = calibrate_matrices_acquisition(slm, stage, hSICtl, hSI, opt)
% Inputs:
%   slm                     SLM object (hardware repo).
%   stage                   stage struct. e.g. a Zaber stage struct.
%   hSICtl                  ScanImage Control handle.
%   hSI                     ScanImage handle.
%   options:
%     stage_distance_um       zaber stage step size in um
%     repetitions             Number of repetitions of each shifting experiment
%     pixels_per_period_slm   Pixels per period of the SLM gradient.
%     backlash_distance_um    Distance to move back and forth to account for backlash in micrometer.
%                           %%%%% Make part of stage struct?
%     stage_settle_time_s     Time in seconds to wait for stage vibrations to decay after moving it.


arguments
    slm (1,1)    SLM
    stage (1,1)
    hSICtl (1,1)
    hSI (1,1)
    opt.stage_distance_um (1,1) = 5                 % Displacement in x and y of zaber stage [um]
    opt.repetitions (1,1) { mustBePositive } = 5    % Number stage shifts to be recorded per axis (average will be computed)
    opt.delta_slope (1,1) = 255/50;                 % slope of the gradient pattern [2*pi/SLM pixels]
    opt.backlash_distance_um (1,1) = 50             % Distance to move back and forth to counteract backlash. See: https://en.wikipedia.org/wiki/Hysteresis#Backlash
    opt.settle_time_s (1,1) { mustBePositive } = 1  % Settle time of vibration [seconds]
end

%% add relevant paths
addpath('C:\git\tpm\calibration\functions\');
import zaber.motion.Units;

%% assertions
assert(sign(opt.stage_distance_um) == sign(opt.backlash_distance_um),...
    'The stage_distance_um must have the same sign as the backlash_distance_um')

%% Construct SLM gradient pattern
bg_patch_id = 1;                            % Gradient Patch ID
rawdata.slm_height_pix = size(slm.getPixels, 2);    % SLM height in pixels
rawdata.slope = [-2,-1,0,1,2]'*opt.delta_slope;
slm_pattern_gradient = rawdata.slope.*(0:rawdata.slm_height_pix);

%% Initializations
rawdata.zoom = hSICtl.hModel.hRoiManager.scanZoomFactor;	% Fetch zoom from ScanImage GUI
rawdata.frame_width_pix = hSICtl.hModel.hRoiManager.linesPerFrame;

rawdata.frames_stage = zeros(rawdata.frame_width_pix, rawdata.frame_width_pix, 2, opt.repetitions+1);
rawdata.frames_gradient = zeros(rawdata.frame_width_pix, rawdata.frame_width_pix, 2, 5);

%% acquire stage images
% Loop over stage axes
for index_stage_axis = 1:length(stage.axes)

    % Move stage back and forth in both axes to account for backlash
    stage.axes(index_stage_axis).moveRelative(-opt.backlash_distance_um, Units.LENGTH_MICROMETRES);
    stage.axes(index_stage_axis).moveRelative( opt.backlash_distance_um, Units.LENGTH_MICROMETRES);
    pause(opt.settle_time_s) %%%% Incorporate as Zaber function wrapper?
    fprintf('\n')

    % Make steps with stage and record frames
    for index_frame = 1:opt.repetitions+1
        rawdata.frames_stage(:, :, index_stage_axis, index_frame) = grabSIFrame(hSI,hSICtl);    % Record frame
        fprintf('Recorded frame %i/%i of stage axis %i\n', index_frame, opt.repetitions+1, index_stage_axis)
    
        stage.axes(index_stage_axis).moveRelative(opt.stage_distance_um, Units.LENGTH_MICROMETRES); % Move stage
        pause(opt.settle_time_s)    % Wait for vibrations to decay
    end
end

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