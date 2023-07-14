function  rawdata_parabola = process_parabola_data_residual(rawdata_parabola)
% To see if there is any shift after the center of the SLM patch is shifted 
% to the center of the back pupil plane. 
% (Using the data acquired from acquire_parabola_data() or from saved raw
% data)
% 
% INPUTS
% rawdata_parabola.frame_ref : initial reference frame (with a blank SLM)
% rawdata_parabola.frame_parabola_shifted : image captured after shifting
%   the SLM patch to match its center with the center of the back pupil
%   plane. 
%
% OUTPUTS
% rawdata_parabola.wk_offset_corrected : cross correlation between reference and
%   shifted frames
% rawdata_parabola.remaining_shift : residual shift (ideally a small number(~1))

%% Use Wiener-Khinchin to calculate the offset
rawdata_parabola.wk_offset_corrected = wiener_khinchin(rawdata_parabola.frame_ref,rawdata_parabola.frame_parabola_shifted);
rawdata_parabola.remaining_shift = calculate_offset_in_peak(rawdata_parabola.wk_offset_corrected); % [TPM frame pixels]
 
end

