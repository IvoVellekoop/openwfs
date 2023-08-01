function Parabola = parabola(ySLM, a)
%% Create parabolic phase pattern on the SLM
% ySLM = number of pixels of the Meadowlark SLM (1920x1152)
% a = parabola parameter (SLMvalues/(SLM pixel size)^2) = 1e-2

% SLM parameters
% pmin = 0;                 % Minimum phase
% pmax = 255;               % Maximum phase

xSLM = ySLM;

x = (1:xSLM)';
y = (1:ySLM);

xcenter = 0.5 * xSLM;
ycenter = 0.5 * ySLM;

% Construct circular pattern coordinates
xcirc = x - xcenter;	% x, with circle center as origin
ycirc = y - ycenter;	% y, with circle center as origin

Parabola = a * (xcirc.^2 + ycirc.^2);
