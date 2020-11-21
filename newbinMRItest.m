function trial = newbinMRItest(subID, sessID, runID)
% function trial = binMRItest(subID, sessID, runID)
% fMRI experiment for BrainImageNet test dataset
% subID, subjet ID, integer[1-20] 
% runID, run ID, integer [1-10] 
% clc;clear;

%% Arguments
% if nargin < 3, subID = 1; end
% if nargin < 2, sessID = 1; end
% if nargin < 1, runID = 1; end

%% Check subject information
% Check subject id
if ~ismember(subID, 1:20), error('subID is a integer within [1:20]!'); end
% Check session id
if ~ismember(sessID, 1:1), error('sessID is a integer within [1:1]!');end
% Check run id
nRun = 10;
if ~ismember(runID, 1:nRun), error('runID is a integer within [1:10]!'); end

%% Data dir 
workDir = pwd;
stimDir = fullfile(workDir,'images');
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


%% Display
imgAngle = 12;
fixOuterAngle = 0.3;
fixInnerAngle = 0.2;
% bkgColor = [128 128 128];
bkgColor = [0.485, 0.456, 0.406] * 255; % ImageNet mean intensity
fixOuterColor = [0 0 0]; % color of fixation circular ring
whiteFixation = [255 255 255]; % color of fixation circular point
redFixation = [255 0 0]; % color of fixation circular point

% compute image pixel
pixelPerMilimeterHor = 1024/390;
pixelPerMilimeterVer = 768/295;
imgPixelHor = pixelPerMilimeterHor * (2 * 1000 * tan(imgAngle/180*pi/2));
imgPixelVer = pixelPerMilimeterVer * (2 * 1000 * tan(imgAngle/180*pi/2));
fixOuterSize = pixelPerMilimeterHor * (2 * 1000 * tan(fixOuterAngle/180*pi/2));
fixInnerSize = pixelPerMilimeterHor * (2 * 1000 * tan(fixInnerAngle/180*pi/2));

%% Response keys setting
PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
startKey = KbName('s');
escKey = KbName('ESCAPE');
cueKey = KbName('1!'); % Left hand:1!,2@

%% Screen setting
Screen('Preference', 'SkipSyncTests', 2);
Screen('Preference','VisualDebugLevel',4);
Screen('Preference','SuppressAllWarnings',1);
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;

%% Create instruction texture
% Makes instruction texture
imgStart = sprintf('%s/%s', 'instruction', 'instructionStartTest.jpg');
imgEnd = sprintf('%s/%s', 'instruction', 'instructionBye.jpg');
startTexture = Screen('MakeTexture', wptr, imread(imgStart));
endTexture = Screen('MakeTexture', wptr, imread(imgEnd));

%% Stimulus
imgName = extractfield(dir(stimDir), 'name');
imgName = imgName(3:end);
nStim = length(imgName);
stimTexture = zeros(nStim,1);
for t = 1:nStim
    img = imread( fullfile(stimDir, imgName{t}));
    img = imresize(img, [imgPixelHor imgPixelVer]);
    stimTexture(t) = Screen('MakeTexture', wptr, img);
end

%% Show instruction
Screen('DrawTexture', wptr, startTexture);
Screen('Flip', wptr);
% Wait ready signal from subject
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(cueKey), break;
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

%% Design
% Collect trial info for this run
% [onset,cond,trueAnswer, key, rt, timingError].  
nTrial = 150; 
trial = zeros(nTrial, 6);

% randomize condition: 1-120 images
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
onset = (0:nTrial-1)*3; 
trial(:,1:2) = [onset',cond']; % [onset, condition]

% True answer for red fix  
trial(cond == 1000,3) = 1;

% End timing of trials
tEnd = trial(:, 1) + 3; 
%% Parms for stimlus presentation and Trials
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 0.5 - 0.5*flipInterval; % on duration for a stimulus
% offDur = 2.5; % off duration for a stimulus
beginDur = 2; % beigining fixation duration
endDur = 2; % ending fixation duration

%% Run experiment
% Show begining fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
tStart = GetSecs;
for t = 1:nTrial
    if trial(t,2) <= 120  % show stimulus with fixation
        Screen('DrawTexture', wptr, stimTexture(trial(t,2)));
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
        
    elseif trial(t,2) == 1000 % show only red fixation
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2  );
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, redFixation , [], 2);
        
    else % show only red fixation
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
    end
    tStim = Screen('Flip',wptr);
    trial(t, 6) = tStim - tStart; % timing error
    
    % Show after stimulus fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
    tFix = Screen('Flip', wptr, tStim + onDur);
    
    % Wait response
    while KbCheck(), end % empty the key buffer
    while GetSecs - tStart < tEnd(t)
        [keyIsDown, tKey, keyCode] = KbCheck();       
        if keyIsDown
            if keyCode(escKey), sca; return;
            elseif keyCode(cueKey),   key = 1;
            else, key = 0; 
            end
            rt = tKey - tFix; % reaction time
            trial(t, 4:5) = [key,rt];
        end
    end
end

% Wait ending fixation
WaitSecs(endDur);

% Show end instruction
Screen('DrawTexture', wptr, endTexture);
Screen('Flip', wptr);
WaitSecs(2);

% Show cursor and close all
ShowCursor;
Screen('CloseAll');

%% Save data for this run
resultFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%2d_run%02d.mat',subID,sessID,runID));
fprintf('Data were saved to: %s\n',resultFile);
save(resultFile,'trial','sessID','subID','runID','imgName');

