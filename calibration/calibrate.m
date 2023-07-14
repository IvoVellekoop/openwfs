%%% Written on 19 may 2022 by Gijs Hannink
%%%
%%% Step 1: add paths and dependencies
%%% Step 2: Setup hardware
%%% Step 3: Acquire data
%%% Step 4: process data
%%% Step 5: save calibration values

%% === Step 1 ===
addpath('C:\git\hardware\matlab');
addpath('C:\git\tpm\setup');
addpath('C:\git\tpm\calibration\functions');
slm_acquisitioni.amplitude = 1e-2;


%% === Step 2 ===
import zaber.motion.Units;      % toolbox: https://www.zaber.com/software/docs/motion-library/ascii/tutorials/install/matlab/
if ~exist('stage', 'var')
    setup_sample_stage
end

if ~exist('slm', 'var')
    [slm, sopt] = SLMsetup();
end

slm.setRect(1,[0 0 1 1]); 
slm.setData(1,0); slm.update;


%% === Step 3 ===
matrices_acquisition = calibrate_matrices_acquisition(slm, stage, hSICtl, hSI);
slm_acquisition = acquire_parabola_data(rawdata_parabola, slm, hSI, hSICtl);


%% === Step 4 ===
[G, M, rawdata_stage] = calibrate_matrices_processing(matrices_acquisition);
[sopt, rawdata_parabola] = process_parabola_data(slm_acquisition, G);


%% == Step 5 ===


