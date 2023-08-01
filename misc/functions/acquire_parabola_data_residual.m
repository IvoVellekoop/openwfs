function rawdata_parabola = acquire_parabola_data_residual(rawdata_parabola, sopt,slm, hSI,hSICtl)
% To acquire the image to compute the residual shift (i.e the shift even after
% the center of the SLM is matched with the center of the back pupil plane). 
%
% INPUTS
% slm:                   the SLM handle
% hSI:                   handle from scanimage
% hSICtl:                handle from scanimage
% sopt.offset_center_slm: normalized (to the length of the 2nd axis of SLM)
%       displacement between the center of the SLM and the center of the
%       back pupil plane
%
% OUTPUTS
% rawdata_parabola.frame_parabola_shifted: image captured after shifting
%   the SLM patch to match its center with the center of the back pupil
%   plane.
%
% STEPS
% 1) Shift the center of the SLM patch to the offset location. 
%       Now the centre of the slm patch will be aligned to the center of the pack
%       pupil plane.
% 2) Project the same parabola, but on the shifted slm pach.
% 2) Get a frame to compute the residual shift.

%% Put shifted parabola pattern on SLM
% The SLM patch is displaced to match its center with the center of the back pupil plane
% (according to sopt.offset_center_slm)
patch_id = 1;
slm.setRect(patch_id, [sopt.offset_center_slm(1) sopt.offset_center_slm(2) 1 1]);
slm.setData(patch_id, rawdata_parabola.Parabola)
slm.update;
 
%% Record frame whilst shifted parabola pattern is on the SLM
rawdata_parabola.frame_parabola_shifted = grabSIFrame(hSI,hSICtl);
end

