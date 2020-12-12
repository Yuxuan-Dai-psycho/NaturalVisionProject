function trial = objectCoCoMemory(subID,sessID,sRun)
% function trial = objectCoCoMemory(subID,sessID,sRun)
% Memory test after BrianImageNet fMRI train experiment
% subID, subjet ID, integer[1-20]
% sessID, session ID, integer [1-4]
% sRun, run ID to start, integer [1-4]
% workdir(or codeDir) -> sitmulus/instruciton/data

if nargin < 3, sRun = 1; end

%% Check subject information+
% Check subject id
if ~ismember(subID, 1:20), error('subID is a integer within [1:20]!'); end
% Check session id
if ~ismember(sessID, 1:4), error('sessID is a integer within [1:4]!');end
% Check start run
if ~ismember(sRun, 1:4), error('sRun is a integer within [1:4]!');end

%% Data dir
workDir = pwd;
trainDir = fullfile(workDir,'data','fmri','test');
sessDir = fullfile(trainDir,sprintf('sub%02d/sess%02d',subID,sessID));
if ~exist(sessDir,'dir'), mkdir(sessDir); end

%% Screen setting
Screen('Preference', 'SkipSyncTests', 1);
Screen('Preference','VisualDebugLevel',4);
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
bkgColor = [0.485, 0.456, 0.406] * 255; % ImageNet mean intensity
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;

%% Response keys setting
% PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
escKey = KbName('ESCAPE');
seenKey = KbName('f'); % F key for left hand
notSeenKey = KbName('j'); % J key for right hand

%% Load stimulus and instruction
imgAngle = 16; fixOuterAngle = 0.3; fixInnerAngle = 0.2;

% Visual angle to pixel
pixelPerMilimeterHor = 1024/390;
pixelPerMilimeterVer = 768/295;
imgPixelHor = round(pixelPerMilimeterHor * (2 * 1000 * tan(imgAngle/180*pi/2)));
imgPixelVer = round(pixelPerMilimeterVer * (2 * 1000 * tan(imgAngle/180*pi/2)));
fixOuterSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixOuterAngle/180*pi/2)));
fixInnerSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixInnerAngle/180*pi/2)));

% Load stimulus
stimDir = fullfile(workDir,'stimulus','test');
expImgName = extractfield(dir(fullfile(stimDir,'expImages')), 'name');
ctrImgName = extractfield(dir(fullfile(stimDir,'ctrImages')), 'name');
imgName = [ctrImgName(3:end),expImgName(3:end)];
nStim = length(imgName);
img = cell(nStim,1);
for t = 1:nStim
    img{t}  = imresize(imread(fullfile(stimDir, imgName{t})),[imgPixelHor imgPixelVer]);
end

% Load instruction image
imgStart = imread(fullfile(workDir, 'instruction', 'testStart.JPG'));
imgEnd = imread(fullfile(workDir, 'instruction', 'testEnd.JPG'));


%% Make design
nTrial = nStim;
imgID = 1:nStim;
cond = ones(nTrial,1); % total trials
cond(1:length(ctrImgName(3:end)),1) = -1;

% [onset,imgID,cond,key,rt,timingError].
trial = zeros(nTrial, 6);
trial(:,2:3) = [imgID,cond]; % [imgID,condition]
% Randomize stimulus
idx = randperm(nStim);
trial = trial(idx,:);


%% Run experiment
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 2.5 - 0.5*flipInterval; % on duration for a stimulus
maskDur = 0.2; % ending duration of each trial
beginDur = 4; % beigining fixation duration
endDur = 4; % ending fixation duration
fixOuterColor = [0 0 0]; % color of fixation circular ring
fixInnerColor = [255 255 255]; % color of fixation circular point


% Show instruction and wait ready signal from subject
startTexture = Screen('MakeTexture', wptr, imgStart);
Screen('DrawTexture', wptr, startTexture);
Screen('Flip', wptr);
Screen('Close',startTexture);
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(seenKey), break; end
end

% Show begining fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor ,[], 2);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus for each trial
tStart = GetSecs;
for t = 1:nTrial
    % Show stimulus with fixation
    stimTexture = Screen('MakeTexture', wptr, img{trial(t,2)});
    Screen('PreloadTextures',wptr,stimTexture);
    Screen('DrawTexture', wptr, stimTexture);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor, [], 2);
    Screen('DrawingFinished',wptr);
    tStim = Screen('Flip',wptr);
    Screen('Close',stimTexture);
    trial(t,1) = tStim - tStart;
    
    % Record response while stimulus is on
    key = 0; rt = 0;
    while KbCheck(), end % empty the key buffer
    while GetSecs - tStim < onDur
        [keyIsDown, tKey, keyCode] = KbCheck();
        if keyIsDown
            if keyCode(escKey),sca; return;
            elseif keyCode(seenKey)
                key = 1; rt = tKey - tStim; break;
            elseif keyCode(notSeenKey)
                key = -1; rt = tKey - tStim;break;
            end
        end
    end
    trial(t, 4:5) = [key,rt];
    
    % Show fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
    Screen('DrawingFinished',wptr);
    tFix = Screen('Flip', wptr);
    while GetSecs - tFix < maskDur, end
end

% Wait ending fixation
endTexture = Screen('MakeTexture', wptr, imgEnd);
Screen('DrawTexture', wptr, endTexture);
Screen('Flip', wptr);
Screen('Close',endTexture);
WaitSecs(endDur);

% Save data for this runID
clear img imgStart imgEnd
resultFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%02d_run%02d_beh.mat',subID,sessID,runID));

% If there is an old file, backup it
if exist(resultFile,'file')
    oldFile = dir(fullfile(sessDir,...
        sprintf('sub%02d_sess%02d_run%02d_beh-*.mat',subID,sessID,runID)));
    % The code works only while try time less than ten
    if isempty(oldFile), n = 1;
    else, n = str2double(oldFile(end).name(end-4)) + 1;
    end
    % Backup the file from last test
    newOldFile = fullfile(sessDir,...
        sprintf('sub%02d_sess%02d_run%02d_beh-%d.mat',subID,sessID,runID,n));
    copyfile(resultFile,newOldFile);
end

% Save data
fprintf('Data were saved to: %s\n',resultFile);
save(resultFile);
% Print sucess info
fprintf('BIN CoCo Memory:sub%d-sess%d-run%d ---- DONE!\n',...
    subID, sessID,runID)
% Show cursor and close all
ShowCursor;
Screen('CloseAll');




