function [G, rawdata_out] = calibrate_matrices_processing(rawdata_in, opt)

arguments
    rawdata_in struct
    opt.stage_distance_um (1,1) = 5                % Displacement in x and y of zaber stage [um]
    opt.delta_slope (1,1) = 255/50;                 % slope of the gradient pattern [2*pi/SLM pixels]
end


[G0, rawdata_out.G0_std] = find_mean_shift(rawdata_in.frames_gradient);

%% Conversion matrix from gradient to angle.

% Conversion matrix for SLM pixels to TPM frame pixels
G = (G0/rawdata_in.zoom).*255/opt.delta_slope;      % [TPM frame pixels*SLM pixels]

