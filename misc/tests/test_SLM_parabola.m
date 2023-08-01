% Code to test the function SLM_parabola()
% C:\Git\tpm\calibration\SLM_parabola.m
%
% This checks whether the process flow is correct and do NOT check the
% validity of the algorithms used.
%
% MAKE SURE
% 1) That you have run setup.m (C:\Git\tpm\setup\setup.m)
%       (for have the variable slm)
% 2) scanimage in virtual TPM (or real two-photon microscope PC)
%       (C:\Git\scanimage\scanimage.m)
%       (for the variables hSI & hSICtl)
% 3) Access to the folder from which data are loaded
%%
clc;

%%
addpath('C:\git\openwfs\calibration');
addpath('C:\git\openwfs\calibration\functions');

%% Loading existing data
% calibration_values.mat
cv = load('P:\TNW\BMPI\Projects\WAVEFRONTSHAPING\data\TPM\4th gen\calibration\archived\220516\calibration_values.mat');
% calibration_values_rawdata.mat
cv_r = load('P:\TNW\BMPI\Projects\WAVEFRONTSHAPING\data\TPM\4th gen\calibration\archived\220516\calibration_values_rawdata.mat');
G = cv.G;

% grabSIFrame = @mock_grabSIFrame
[sopt, rawdata]= SLM_parabola(G, slm, hSI, hSICtl)

% function return_image = mock_grabSIFrame(hSI, hSICtl)
%     return_image = rand(10);
% end

clear slm;




