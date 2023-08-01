function [sopt, rawdata_parabola] = process_parabola_data(rawdata_parabola, G)
% Function to determine the shift between the center of the SLM patch and 
% the center of the back pupil plane. 
% (Using the data acquired from acquire_parabola_data() or from saved raw
% data)
%
% INPUTS
% rawdata_parabola.frame_ref : reference frame (with a blank SLM)
% rawdata_parabola.frame_Sample_shifted : Frame after the putting the parabola on SLM
% G : Matrix to convert from SLM gradients [1/SLMpixels] to Galvo tilts [TPM image pixels]
% rawdata_parabola.zoom_parabola : Zoom use in the scan image
%
% OUTPUTS
% rawdata_parabola.wk_offset : cross correlation between reference and
%   shifted frames
% rawdata_parabola.image_shift : shift (in pixels) between the frames
%       It will be a vector of two elements, corresponding to the
%       shifts along first and second axis, respectively.
% rawdata_parabola.d_slm : corresponding displacement on the SLM (pixels)
% sopt.offset_center_slm : normalized displacement 
%   (normalized to the shortest side (2nd axis) of the slm)


%% Use Wiener-Khinchin to calculate the offset
rawdata_parabola.wk_offset = wiener_khinchin(rawdata_parabola.frame_ref,rawdata_parabola.frame_Sample_shifted);
rawdata_parabola.image_shift = calculate_offset_in_peak(rawdata_parabola.wk_offset); % [TPM frame pixels]
 
%% Calculate the offset on the SLM (in SLM pixels)
% multiply TPM image offset with G matrix:
b_g = G^(-1)*(rawdata_parabola.image_shift/rawdata_parabola.zoom_parabola)';  % [1/SLM pixels]
b = b_g*(256);  % [SLM values/SLM pixels]

% displacement on the SLM
rawdata_parabola.d_slm = (b)/(2*rawdata_parabola.amplitude);  % [SLM pixels]
sopt.offset_center_slm = rawdata_parabola.d_slm/rawdata_parabola.ySLM;
end

