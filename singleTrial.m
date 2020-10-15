function response = singleTrial(wptr, rect, stimTexture)
% Define the keyboard and present images for single trail
% F - like - 1  J - dislike - 0  None response - -1

%% Prepare parameters for time and location
stimulus_onset = 2;
blank_Interval = 2;
fixSize = 10;
fixColor = [255 255 255];
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels

%% Response keys setting 
KbName('UnifyKeyNames');
likeKey = KbName('f');
disLikeKey = KbName('j'); % stop and exit
escKey   = KbName('escape'); % stop and exit
    
%% Show the corresponding stimuli
Screen('DrawTexture', wptr, stimTexture);
Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
Screen('Flip',wptr);

%% Run in single trail
t0 = GetSecs;   
while true     
    [~, ~, key_Code] = KbCheck;      
    % get response
    if key_Code(likeKey)
        response = 1;
    elseif key_Code(disLikeKey)
        response = 0;
    else 
        response = -1;
    end
    % using in debug
    if key_Code(escKey) 
        response = 'break';
        break;
    end
    % break after onset time
    tt = GetSecs;   
    if tt-t0>stimulus_onset
        break
    end
end

%% Show the fixation  
Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
Screen('Flip', wptr);
WaitSecs(blank_Interval);    

end