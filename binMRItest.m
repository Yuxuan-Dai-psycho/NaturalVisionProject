function trial = binMRItest(subID, sessID, runID)
% function trial = binMRItest(subID, sessID, runID)
% fMRI experiment for BrainImageNet test dataset
% subID, subjet ID, integer[1-20]
% runID, run ID, integer [1-10]
% workdir(or codeDir) -> sitmulus/instruciton/data 


%% Check subject information
% Check subject id
if ~ismember(subID, 1:20), error('subID is a integer within [1:20]!'); end
% Check session id
if ~ismember(sessID, 1:1), error('sessID is a integer within [1:1]!');end
% Check run id
nRun = 10;
if ~ismember(runID, 1:nRun), error('runID is a integer within [1:10]!'); end

%% Data dir
% Make work dir
workDir = pwd;

% Make data dir
dataDir = fullfile(workDir,'data');
if ~exist(dataDir,'dir'), mkdir(dataDir), end

% Make fmri dir
mriDir = fullfile(dataDir,'fmri');
if ~exist(mriDir,'dir'), mkdir(mriDir), end

% Make test dir for the subject
testDir = fullfile(mriDir,'test');
if ~exist(testDir,'dir'), mkdir(testDir),end

% Make subject dir
subDir = fullfile(testDir,sprintf('sub%02d', subID));
if ~exist(subDir,'dir'), mkdir(subDir),end

% Make session dir
sessDir = fullfile(subDir,sprintf('sess%02d', sessID));
if ~exist(sessDir,'dir'), mkdir(sessDir), end

%% Screen setting
Screen('Preference', 'SkipSyncTests', 1);
% Screen('Preference','VisualDebugLevel',4);
% Screen('Preference','SuppressAllWarnings',1);
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
bkgColor = [0.485, 0.456, 0.406] * 255; % ImageNet mean intensity
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;

%% Response keys setting
% PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
startKey = KbName('s');
escKey = KbName('ESCAPE');
cueKey1 = KbName('1!'); % Left hand:1!
cueKey2 = KbName('2@'); % Left hand:2@

%% Load stimulus and instruction
imgAngle = 16;
fixOuterAngle = 0.2;% 0.3
fixInnerAngle = 0.1;% 0.2
fixOuterColor = [0 0 0]; % color of fixation circular ring
whiteFixation = [255 255 255]; % color of fixation circular point
redFixation = [255 0 0]; % color of fixation circular point

% Visual angle to pixel
pixelPerMilimeterHor = 1024/390;
pixelPerMilimeterVer = 768/295;
imgPixelHor = round(pixelPerMilimeterHor * (2 * 1000 * tan(imgAngle/180*pi/2)));
imgPixelVer = round(pixelPerMilimeterVer * (2 * 1000 * tan(imgAngle/180*pi/2)));
fixOuterSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixOuterAngle/180*pi/2)));
fixInnerSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixInnerAngle/180*pi/2)));

% Load stimulus
stimDir = fullfile(workDir,'stimulus','test','images');
imgName = extractfield(dir(stimDir), 'name');
imgName = imgName(3:end);
nStim = length(imgName);
img = cell(nStim,1);
for t = 1:nStim
    img{t}  = imresize(imread(fullfile(stimDir, imgName{t})), [imgPixelHor imgPixelVer]);
end

% Load instruction image
imgStart = imread(fullfile(workDir, 'instruction', 'testStart.JPG'));
imgEnd = imread(fullfile(workDir, 'instruction', 'testEnd.JPG'));

%% Show instruction
startTexture = Screen('MakeTexture', wptr, imgStart);
Screen('DrawTexture', wptr, startTexture);
Screen('Flip', wptr);
Screen('Close',startTexture);

% Wait ready signal from subject
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && (keyCode(cueKey1) || keyCode(cueKey2)), break;
    end
end
% Show ready signal
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, redFixation, [], 2);
Screen('Flip', wptr);

% Wait trigger(S key) to begin the test
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(startKey), break;
    elseif keyIsDown && keyCode(escKey), sca; return;
    end
end

%% Make design
% [onset,cond,trueAnswer, key, rt, timingError].
nTrial = 150;
trial = zeros(nTrial, 6);

% Randomize condition: 1-120 images
cond = 1:nTrial;% total trials
cond(121:135) = 1000; % Null trials with red fix
cond(136:nTrial) = 2000; % Null trials with white fix
while true
    cond = cond(randperm(length(cond)));
    tmp = diff(cond);
    tmp(abs(tmp)==1000) = 0 ; % null trials can not be colse each other
    if cond(1) > 120, tmp(1) = 0; end  % the first trial can not be null trial
    if all(tmp),break; end
end
onset = (0:nTrial-1)*3;% each trial last 3s
trial(:,1:2) = [onset',cond']; % [onset, condition]

% True answer for red fix
trial(cond == 1000,3) = 1;

% End timing of trials
tEnd = trial(:, 1) + 3;

%% Run experiment
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 0.5 - 0.5*flipInterval; % on duration for a stimulus
beginDur = 16; % beigining fixation duration
endDur = 16; % ending fixation duration

% Show begining fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
tStart = GetSecs;
for t = 1:nTrial
    if trial(t,2) <= 120  % show stimulus with fixation
        stimTexture = Screen('MakeTexture', wptr, img{trial(t,2)});
        Screen('DrawTexture', wptr, stimTexture);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation ,[], 2);
        Screen('DrawingFinished',wptr);
        Screen('Close',stimTexture);
        
    elseif trial(t,2) == 1000 % show only red fixation
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2  );
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, redFixation , [], 2);
        Screen('DrawingFinished',wptr);
        
    else % show only white fixation
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
        Screen('DrawingFinished',wptr);
    end
    tStim = Screen('Flip',wptr);
    trial(t, 6) = tStim - tStart; % timing error
    
    
    % If subject respond in stimulus presenting, we record it
    key = 0; rt = 0;
    while KbCheck(), end % empty the key buffer
    while GetSecs - tStim < onDur
        [keyIsDown, tKey, keyCode] = KbCheck();
        if keyIsDown
            if keyCode(escKey),sca; return;
            elseif keyCode(cueKey1) || keyCode(cueKey2)
                key = 1; rt = tKey - tStim;
            end
        end
    end
    
    % Show after stimulus fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
    Screen('DrawingFinished',wptr);
    Screen('Flip',wptr);

    % If subject have ready responded in stimtulus presenting, we'll not
    % record it in fixation period; if not, we record it.
    if rt
        while KbCheck(), end
        while GetSecs - tStart < tEnd(t)
            [keyIsDown, ~, keyCode] = KbCheck();
            if keyIsDown && keyCode(escKey), sca; return; end
        end
    else
        while KbCheck(), end 
        while GetSecs - tStart < tEnd(t)
            [keyIsDown, tKey, keyCode] = KbCheck();
            if keyIsDown
                if keyCode(escKey),sca; return;
                elseif keyCode(cueKey1) || keyCode(cueKey2)
                    key = 1; rt = tKey - tStim;
                end
            end
        end
    end
    trial(t, 4:5) = [key, rt];
end

% Wait ending fixation
WaitSecs(endDur);

% Show end instruction
endTexture = Screen('MakeTexture', wptr, imgEnd);
Screen('PreloadTextures',wptr,endTexture);
Screen('DrawTexture', wptr, endTexture);
Screen('DrawingFinished',wptr);
Screen('Flip', wptr);
Screen('Close',endTexture); 
WaitSecs(2);

% Show cursor and close all
ShowCursor;
Screen('CloseAll');

%% Save data for this run
clear img imgStart imgEnd
resultFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%2d_run%02d.mat',subID,sessID,runID));
fprintf('Data were saved to: %s\n',resultFile);
save(resultFile);
% Print sucess info
fprintf('BINtest subID:%d, sessID:%d, runID:%d ---- DONE!', subID, sessID,runID)


