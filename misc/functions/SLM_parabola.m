function [sopt,rawdata_parabola] = SLM_parabola(G, slm, hSI, hSICtl, amplitude, plotfigs)
% Calibration to align the center an SLM patch to the center of the back 
% pupil plane. 
% 
% From this shift we calculate which way the pattern on the SLM must be
% shifted to align the centers.
% The desired OUTPUT values of this script are the rawdata and:
% sopt.offset_center_slm
% 
% 
% INPUT
% G:                     Matrix to convert from SLM gradients [1/SLMpixels]
%                        to Galvo tilts [TPM image pixels]
% slm:                   the SLM handle
% hSI:                   handle from scanimage
% hSICtl:                handle from scanimage
% amplitude:             amplitude        parabola parameter, set to 1e-2 as default
% plotfigs:              returns the figures if true
%
% REQUIREMENTS
% C:\git\scanimage\scanimage.m must be running 

arguments
    G (2,2) 
    slm SLM
    hSI 
    hSICtl 
    amplitude (1,1) = 1e-2
    plotfigs logical = false
end

rawdata_parabola.amplitude = amplitude;

% Obtaining the rawdata for parabola calibration (aligning center of SLM patch 
% to the center of the back pupil plane). Promarily, a reference (1st) image 
% and a defocused (2nd) image.
rawdata_parabola = acquire_parabola_data(rawdata_parabola, slm, hSI, hSICtl);

% Processing the rawdata (reference image, defocused images, G and Zoom).
% Primarily to obtain sopt.offset_center_slm : displacements (along the 2 axes) 
% of center of SLM patch & the center of the back pupil plane.
[sopt, rawdata_parabola] = process_parabola_data(rawdata_parabola, G);

% Obtaining the rawdata to compute the residual shift (shift that is remaining 
% after the slm patch is shifted to the center of the back pupil plane).
% Primarily, a (3rd) image after putting the same parabolic phase pattern, but
% on a shifted SLM patch
rawdata_parabola = acquire_parabola_data_residual(rawdata_parabola, sopt,slm, hSI,hSICtl);

% Compute the shift between the reference (1st) image and the image with
% the parabola shifted (3rd).
rawdata_parabola = process_parabola_data_residual(rawdata_parabola);

%% Plot figures
if plotfigs
    figure(1); imagesc(rawdata_parabola.wk_offset); axis image;
    title('Wiener-Khinchin of frame\_ref - frame\_sample\_shifted');
    figure(2); imagesc(rawdata_parabola.frame_ref); axis image; title('frame\_ref');
    figure(3); imagesc(rawdata_parabola.frame_Sample_shifted); axis image;
    title('frame\_Sample\_shifted');
    figure(4); imagesc(rawdata_parabola.frame_parabola_shifted); axis image;
    title('frame\_parabola\_shifted');
    figure(6); imagesc(rawdata_parabola.wk_offset_corrected); axis image;
    title('Wiener-Khinchin of frame\_ref - frame\_sample\_shifted');
end