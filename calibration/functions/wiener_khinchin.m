function y = wiener_khinchin(F,G)
% calculate the "cross correlation" suitable (only) for calculating shifts
% between two images F and G using Wiener-Khinchin theorem.
% When F = G, peak will be at (1,1)

% BEWARE
% 1) Doesnot work for matrices of different dimensions 
%   (whereas cross correlation does work)
% 2) Dimensions are not matching: the dimension of cross correlation between 
%       an mxn matrix and a pxq matrix has  to be (m+p)x(n+q)
% 3) Not normalized. (Fix the issues 1&2 before this)

y = ifft2(conj(fft2((F-mean2(F)))) .* fft2(G-mean2(G)) ) /(norm(F)*norm(G));