function trial = objectMRItrain(subID,sessID,runID)
% function [subject,task] = objectMRItrain(subID,sessID,runID)
% Brain ImageNet fMRI experiment stimulus procedure
% subject do animate vs. inanimate discrimination task
% subID, subjet ID, integer[1-20]
% sessID, session ID, integer [1-4]
% runID, run ID, integer [1-10]
% workdir(or codeDir) -> sitmulus/instruciton/data 

%% Check subject information
% Check subject id
if ~ismember(subID, 1:20), error('subID is a integer within [1:20]!'); end
% Check session id
if ~ismember(sessID, 1:4), error('sessID is a integer within [1:4]!');end
% Check run id
if ~ismember(runID, 1:10), error('runID is a integer within [1:10]!'); end
nRun = 10;

%% Data dir
% Make work dir
workDir = pwd;

% Make data dir
dataDir = fullfile(workDir,'data');
if ~exist(dataDir,'dir'), mkdir(dataDir), end

% Make fmri dir
mriDir = fullfile(dataDir,'fmri');
if ~exist(mriDir,'dir'), mkdir(mriDir), end

% Make train dir
trainDir = fullfile(mriDir,'train');
if ~exist(trainDir,'dir'), mkdir(trainDir),end

% Make subject dir
subDir = fullfile(trainDir,sprintf('sub%02d', subID));
if ~exist(subDir,'dir'), mkdir(subDir),end

% Make session dir
sessDir = fullfile(subDir,sprintf('sess%02d', sessID));
if ~exist(sessDir,'dir'), mkdir(sessDir), end

%% Screen setting
Screen('Preference', 'SkipSyncTests', 1);
% Screen('Preference','VisualDebugLevel',4);
% Screen('Preference','SuppressAllWarnings',1);
bkgColor = [0.485, 0.456, 0.406] * 255; % ImageNet mean intensity
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;

%% Response keys setting
% PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
startKey = KbName('s');
escKey = KbName('ESCAPE');

% Left hand for animate and right hand for inanimate
animateKey1 = KbName('1!'); % Left hand:1!
animateKey2 = KbName('2@'); % Left hand:2@
inanimateKey1 = KbName('3#'); % Right hand: 3#
inanimateKey2 = KbName('4$'); % Right hand: 4$

%% Make design for this session
% Set design dir
designDir = fullfile(workDir,'stimulus','train','designMatrix');
designFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%02d_design.mat',subID,sessID));
if ~exist(designFile,'file')
    load(fullfile(designDir,'BIN.mat'),'BIN');
    sess = 4*(subID-1)+ sessID;
    
    sessPar = squeeze(BIN.paradigmClass(:,sess,:));
    classOrder = sessPar(:,2);
    sessStim = BIN.stimulus(classOrder,sess);
    sessStim = reshape(sessStim,[100,nRun]);
    sessClass = reshape(BIN.classID(classOrder), [100,nRun]);
    sessPar = reshape(sessPar,[100,nRun,3]);
    save(designFile,'sessStim','sessPar','sessClass');
end

% Load session design
load(designFile,'sessStim','sessPar','sessClass');

% Image for this run
runStim = sessStim(:,runID);
runClass = sessClass(:,runID);

% Collect trial info for this run
nStim = length(runStim);
nTrial = nStim;
trial = zeros(nTrial, 6); % [onset, class, dur, key, RT, timing error]
trial(:,1:3) = squeeze(sessPar(:,runID,:)); % % [onset, class, dur]

%% Load stimulus and instruction
% Visule angle for stimlus and fixation
imgAngle = 16;
fixOuterAngle = 0.2;
fixInnerAngle = 0.1;

% Visual angle to pixel
pixelPerMilimeterHor = 1024/390;
pixelPerMilimeterVer = 768/295;
imgPixelHor = round(pixelPerMilimeterHor * (2 * 1000 * tan(imgAngle/180*pi/2)));
imgPixelVer = round(pixelPerMilimeterVer * (2 * 1000 * tan(imgAngle/180*pi/2)));
fixOuterSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixOuterAngle/180*pi/2)));
fixInnerSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixInnerAngle/180*pi/2)));

% Load stimuli
stimDir = fullfile(workDir,'stimulus','train','images');
img = cell(nStim,1);
for t = 1:nStim
    imgFile = fullfile(stimDir, runClass{t}, runStim{t});
    img{t} = imresize(imread(imgFile), [imgPixelHor imgPixelVer]);
end

% Load  instruction
imgStart = imread(fullfile(workDir, 'instruction', 'trainStart.JPG'));
imgEnd = imread(fullfile(workDir, 'instruction', 'trainEnd.JPG'));

%% Show instruction
startTexture = Screen('MakeTexture', wptr, imgStart);
Screen('PreloadTextures',wptr,startTexture);
Screen('DrawTexture', wptr, startTexture);
Screen('DrawingFinished',wptr);
Screen('Flip', wptr);
Screen('Close',startTexture); 

% Wait ready signal from subject
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && (keyCode(animateKey1) || keyCode(animateKey2)), break;
    end
end
readyDotColor = [255 0 0];
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, readyDotColor, [], 2);
Screen('DrawingFinished',wptr);
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
onDur = 1 - 0.5*flipInterval; % on duration for a stimulus
runDur = 480; % duration for a run
beginDur = 16; % beigining fixation duration
endDur = 16; % ending fixation duration
fixOuterColor = [0 0 0]; % color of fixation circular ring
fixInnerColor = [255 255 255]; % color of fixation circular point
tEnd = [trial(2:end, 1);runDur]; % make sequence of tEnd

% Show begining fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor ,[], 2);
Screen('DrawingFinished',wptr);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
tStart = GetSecs;
for t = 1:nTrial
    % Show stimulus with fixation
    stimTexture = Screen('MakeTexture', wptr, img{t});
    Screen('PreloadTextures',wptr,stimTexture);
    Screen('DrawTexture', wptr, stimTexture);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor, [], 2);
    Screen('DrawingFinished',wptr);
    tStim = Screen('Flip',wptr);
    Screen('Close',stimTexture);
    trial(t, 6) = tStim - tStart; % timing error
    
    % If subject responds in stimulus presenting, we record it
    key = 0; rt = 0;
    while KbCheck(), end % empty the key buffer
    while GetSecs - tStim < onDur
        [keyIsDown, tKey, keyCode] = KbCheck();
        if keyIsDown
            if keyCode(escKey),sca; return;
            elseif keyCode(animateKey1) || keyCode(animateKey2)
                key = 1; rt = tKey - tStim;
            elseif keyCode(inanimateKey1)|| keyCode(inanimateKey2)
                key = -1; rt = tKey - tStim;
            end
        end
    end
  
    % Show fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor ,[], 2);
    Screen('DrawingFinished',wptr);
    Screen('Flip', wptr);
    
    % If subject has ready responded in stimtulus presenting, we'll not
    % record it in fixation period; if not, we record it.
    if rt
        while GetSecs - tStart < tEnd(t)
            [keyIsDown, ~, keyCode] = KbCheck();
            if keyIsDown && keyCode(escKey), sca; return; end
        end
    else
        while GetSecs - tStart < tEnd(t)
            [keyIsDown, tKey, keyCode] = KbCheck();
            if keyIsDown
                if keyCode(escKey),sca; return;
                elseif keyCode(animateKey1) || keyCode(animateKey2)
                    key = 1; rt = tKey - tStim;
                elseif keyCode(inanimateKey1)|| keyCode(inanimateKey2)
                    key = -1; rt = tKey - tStim;
                end
            end
        end
    end
    trial(t, 4:5) = [key,rt];
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

%% Evaluate the response
load(fullfile(designDir,'animate_or_not.mat'),'animate_label');
% trial, nTial * 6 array;  % [onset, class, dur, key, RT, timing error]
% Make target matrix nTrial x nCond
target = zeros(nTrial,2);
animate_label = animate_label(trial(:,2));
target(:,1) = animate_label == 1;
target(:,2) = animate_label == -1;

% Make response matrix nTrial x nCond
response = zeros(nTrial,2);
response(:,1) = trial(:,4) == 1;
response(:,2) = trial(:,4) == -1;

% Summarize the response with figure 
binResponseEvaluation(target, response,{'Animate', 'Inanimate'});

% Save figure
figureFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%02d_run%02d.jpg',subID,sessID,runID));
print(figureFile,'-djpeg');

%% Save data for this run
clear img imgStart imgEnd
resultFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%02d_run%02d.mat',subID,sessID,runID));

% If there is an old file, backup it
if exist(resultFile,'file')
    oldFile = dir(fullfile(sessDir,...
        sprintf('sub%02d_sess%02d_run%02d-*.mat',subID,sessID,runID)));
    
    % The code works only while try time less than ten
    if isempty(oldFile), n = 1;
    else, n = str2double(oldFile(end).name(end-4)) + 1;
    end
    
    % Backup the file from last test 
    newOldFile = fullfile(sessDir,...
        sprintf('sub%02d_sess%02d_run%02d-%d.mat',subID,sessID,runID,n));
    copyfile(resultFile,newOldFile);
end

% Save file
fprintf('Data were saved to: %s\n',resultFile);
save(resultFile);

% Print sucess info
fprintf('BINtrain subID:%d, sessID:%d, runID:%d ---- DONE!', subID, sessID,runID)