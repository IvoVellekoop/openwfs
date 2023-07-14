function grabFrame = grabSIFrame(hSI, hSICtl, n_frames)
% GRABSFRAME(HSI, HSICTL, N_FRAMES) grabs a frame from the ScanImage
%   software
% hSI and hSICtl are the handles of the ScanImage software components
% n_frames is the number of frames to sum together (defaults to 1)
%
% Note: ScanImage will be in 'idle' mode after calling this function.


    if (nargin < 3)
        n_frames = 1;
    end
    if ~strcmp(hSI.acqState, 'idle')
        disp('Scanimage not idle. Trying to abort...')
        hSICtl.abortButton
        pause(0.2)
        if ~strcmp(hSI.acqState, 'idle')
            error('Could not abort current scanimage operation')
        end
        disp('Succesfully aborted current operation.')
    end

    for n=1:n_frames
        hSICtl.grabButton 
        if ~strcmp(hSI.acqState,'grab')                             % make sure scanimage is in grab mode
            error('scan image needs to run in grab mode to acquire feedback');
        end  
        while strcmpi(hSI.acqState,'grab')  % make sure scanimage is in idle state anymore
            pause(0.001);
        end
        if n == 1
            grabFrame = single(hSI.hDisplay.lastFrame{1}); 
        else
            grabFrame = grabFrame + single(hSI.hDisplay.lastFrame{1});
        end
    end
end