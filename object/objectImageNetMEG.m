function trial = objectImageNetMEG(subID,sessID,runID)
% function [subject,task] = objectImageNetMEG(subID,sessID,runID)
% Brain ImageNet MEG experiment stimulus procedure
% subID, subjet ID, integer[1-20] 
% sessID, session ID, integer [1-4]
% runID, run ID, integer [1-10] 

%% Check subject information
% Check subject id
if ~ismember(subID, 1:20), error('subID is a integer within [1:20]!'); end
% Check session id
if ~ismember(sessID, 1:2), error('sessID is a integer within [1:2]!');end
% Check run id
if ~ismember(runID, 1:10), error('runID is a integer within [1:10]!'); end

%% Data dir 
  = './stim';
stimDir = fullfile(workDir,'images');
designDir = fullfile(workDir,'MEGDesignMatrix');

% Make data dir
dataDir = fullfile(workDir,'data');
if ~exist(dataDir,'dir'), mkdir(dataDir), end

% Make MEG dir
megDir = fullfile(dataDir,'MEG');
if ~exist(megDir,'dir'), mkdir(megDir), end

% Make train dir 
trainDir = fullfile(megDir,'train');
if ~exist(trainDir,'dir'), mkdir(trainDir),end

% Make subject dir
subDir = fullfile(trainDir,sprintf('sub%02d', subID));
if ~exist(subDir,'dir'), mkdir(subDir),end

% Make session dir
sessDir = fullfile(subDir,sprintf('sess%02d', sessID));
if ~exist(sessDir,'dir'), mkdir(sessDir), end

%% Stimulus for this sess
designFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%02d_design.mat',subID,sessID));
if ~exist(designFile,'file')
    load(fullfile(designDir,'MEGDesignMatrix.mat'), 'MEGDesignMatrix');
    sess = 2*(subID-1)+ sessID;
    sessStim = MEGDesignMatrix.stim(sess, :, :);
    sessClass = MEGDesignMatrix.class(sess, :, :);
    sessFixationTime = MEGDesignMatrix.fixationTime(sess, :, :);
    sessAccumTime = MEGDesignMatrix.accumTime(sess, :, :);
    sessType = MEGDesignMatrix.type(sess, :, :);
    sessStim = squeeze(sessStim); sessClass = squeeze(sessClass); sessType = squeeze(sessType);
    sessFixationTime = squeeze(sessFixationTime); sessAccumTime = squeeze(sessAccumTime);
    save(designFile, 'sessStim', 'sessClass', 'sessFixationTime', 'sessAccumTime', 'sessType');
end

% Load session design
load(designFile, 'sessStim', 'sessClass', 'sessFixationTime', 'sessAccumTime', 'sessType');

% Image for this run
runStim = sessStim(runID, :);
runClass = sessClass(runID, :);
runFixationTime = sessFixationTime(runID, :);
runType = sessType(runID, :);
tEnd = sessAccumTime(runID, :);

% Collect trial info for this run 
nStim = length(runStim);
nTrial = nStim;
trial = zeros(nTrial, 6); % [onset, class, dur, key, rt]
trial(:,1) = tEnd;

%% Prepare params
imgAngle = 12;
fixOuterAngle = 0.3;
fixInnerAngle = 0.2;
readyDotColor = [255 0 0];
% bkgColor = [128 128 128];
bkgColor = [0.485, 0.456, 0.406] * 255; % ImageNet mean intensity

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
likeKey = KbName('1!'); % Left hand:1!,2@
disLikeKey = KbName('3#'); % Right hand: 3#,4$ 

%% Screen setting
Screen('Preference', 'SkipSyncTests', 2);
Screen('Preference','VisualDebugLevel',4);
Screen('Preference','SuppressAllWarnings',1);
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;
Priority(2);

%% Create instruction texture
% Makes instruction texture
imgStart = sprintf('%s/%s', 'instruction', 'instructionStartTrain.jpg');
imgEnd = sprintf('%s/%s', 'instruction', 'instructionBye.jpg');
startTexture = Screen('MakeTexture', wptr, imread(imgStart));
endTexture = Screen('MakeTexture', wptr, imread(imgEnd));

%% Make stimuli texture
stimTexture = zeros(nStim,1);
for t = 1:nStim
    imgFile = fullfile(stimDir, runClass{t}, runStim{t});
    img = imread(imgFile);
    img = imresize(img, [imgPixelHor imgPixelVer]);
    stimTexture(t) = Screen('MakeTexture', wptr, img);
end

%% Show instruction
Screen('DrawTexture', wptr, startTexture);
Screen('Flip', wptr);
% Wait ready signal from subjectsss
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(likeKey), break;
    end
end

Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, readyDotColor, [], 2);
Screen('Flip', wptr);

% Wait trigger(S key) to begin the test
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(startKey), break;
    elseif keyIsDown && keyCode(escKey), sca; return;
    end
end

%% Run experiment
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 0.5 - 0.5*flipInterval; % on duration for a stimulus
offDur = runFixationTime; % off duration for a stimulus
runDur = 570; % duration for a run
beginDur = 6; % beigining fixation duration
endDur = 6; % ending fixation duration
fixOuterColor = [0 0 0]; % color of fixation circular ring
fixInnerColor = [255 255 255]; % color of fixation circular point
% tEnd = [trial(2:end, 1);runDur]; % make sequence of tEnd

% Show begining fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
tStart = GetSecs;
for t = 1:nTrial
    % Show stimulus with fixation
    Screen('DrawTexture', wptr, stimTexture(t));
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
    tStim = Screen('Flip',wptr);
    trial(t, 6) = tStim - tStart; % timing error
    
    while KbCheck(), end % empty the key buffer
    while ~runType(t) && (GetSecs - tStim < onDur)
        [keyIsDown, tKey, keyCode] = KbCheck();       
        if keyIsDown
            if keyCode(escKey), sca; return;
            elseif keyCode(likeKey),    key = 1;
            elseif keyCode(disLikeKey), key = -1;
            else,  key = 0;  
            end
            rt = tKey - tStim; % reaction time
            trial(t, 4:5) = [key,rt];
            break;
        end
    end
    
    % Show begining fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
    tFix = Screen('Flip', wptr, tStim + onDur);
    
    while KbCheck(), end % empty the key buffer
    while ~runType(t) && (GetSecs - tFix < offDur(t))
        [keyIsDown, tKey, keyCode] = KbCheck();       
        if keyIsDown
            if keyCode(escKey), sca; return;
            elseif keyCode(likeKey),    key = 1;
            elseif keyCode(disLikeKey), key = -1;
            else,  key = 0;  
            end
            rt = tKey - tFix; % reaction time
            trial(t, 4:5) = [key,rt];
            break;
        end
    end
    
    % wait until tEnd
    while GetSecs - tStart < tEnd(t)
        [~, ~, keyCode] = KbCheck();
        if keyCode(escKey), sca; return; end
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
    sprintf('sub%02d_sess%02d_run%02d.mat',subID,sessID,runID));
fprintf('Data were saved to: %s\n',resultFile);
save(resultFile,'trial','sessID','subID','runID');