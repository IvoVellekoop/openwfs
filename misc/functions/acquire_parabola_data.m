function rawdata_parabola = acquire_parabola_data(rawdata_parabola, slm, hSI, hSICtl)
% function to acquire the raw data required to align the center of SLM patch 
% to the center of back pupil plane.
%
% INPUT
% rawdata_parabola.amplitude : amplitude of the parabola
% slm:                   the SLM handle
% hSI:                   handle from scanimage
% hSICtl:                handle from scanimage
% 
% OUTPUTS
% rawdata_parabola.frame_ref : reference frame (with a blank SLM)
% rawdata_parabola.Parabola : Parabola that was put on SLM
% rawdata_parabola.frame_Sample_shifted : Frame after the putting the parabola on SLM
%                   and compensating for the defocus caused by the parabola 
%                   (by increasing the distance between sample and the objective)
% rawdata_parabola.ySLM : number of pixels along the 2nd axis of the SLM
%   Eg: a double, 1152 for Meadowlark SLM having a resolution 1920x1152 pixels
%
% STEPS
% 1) Take a reference image. 
% 2) Put a parabolic phase pattern on SLM.
% 3) Increase the distance between objective and sample to compensate the
%   defocus. Done manually. 
% 4) Take the 2nd image (shifted version of reference image)

%% Fetch zoom from ScanImage
rawdata_parabola.zoom_parabola = hSICtl.hModel.hRoiManager.scanZoomFactor;   

%% Record reference frame
% Put blank pattern on SLM
patch_id = 1;
slm.setRect(patch_id, [0 0 1 1]);
slm.setData(patch_id, 0)
slm.update;
% record a reference frame
rawdata_parabola.frame_ref = grabSIFrame(hSI,hSICtl);   

%% Create parabolic pattern for on the SLM
% creating a parabola with a resolution that of the short side of the SLM
rawdata_parabola.ySLM = size(slm.getPixels,2);   % number of pixels of the Meadowlark SLM (1920x1152) [SLM pixels]
rawdata_parabola.Parabola = parabola(rawdata_parabola.ySLM,rawdata_parabola.amplitude);

% Put parabola pattern on SLM
slm.setRect(patch_id, [0 0 1 1]);
slm.setData(patch_id, rawdata_parabola.Parabola)
slm.update;
 
%% move the top objective about 36 um upwards to record the shifted frame.
% This is necessairy because the parabola acts as a "lens" and shifts the
% focal plane of the system in axial direction.
disp('Move the top objective about 36 um upwards to record the shifted frame and press Enter');
pause

%% Record shifted frame whilst parabola pattern is on the SLM
rawdata_parabola.frame_Sample_shifted = grabSIFrame(hSI,hSICtl);
end

