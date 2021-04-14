function trial = ImageNetMEG(subID,runID)
% function [subject,task] = ImageNetMEG(subID,sessID,runID)
% ImageNet MEG experiment stimulus procedure
% subjects perform animate vs. inanimate discrimination task
% subID, subjet ID, integer[1-30]
% sessID, session ID, integer [1-4]
% runID, run ID, integer [1-5]
% workDir(or codeDir) -> sitmulus/instruciton/data 
% ImageNet MEG has two rounds of exp. 
% Two sessions  will be conducted in each round.

%% Check subject information
% Check subject id
if ~ismember(subID, 1:30), error('subID is a integer within [1:30]!'); end
% Check session id
% if subID <= 10
%     if ~ismember(sessID, 1:4), error('sessID can be [1:4] for SubID 1-10!');end
% else
%     if ~ismember(sessID, 1:2), error('sessID can only be [1:2] for SubID 11-30!');end
% end
% Check run id
if subID < 10
    if ~ismember(runID, 1:20), error('runID is a integer within [1:20] for SubID 1-10!'); end
else
    if ~ismember(runID, 1:10), error('runID is a integer within [1:10] for SubID 11-30!'); end
end
nRun = 5;
sessID = floor(runID/nRun);
%% Data dir
% Check workDir for MEG test 
workDir = pwd;
fprintf('MEG ImageNet workDir is:\n%s\n',workDir)

% Make data dir
dataDir = fullfile(workDir,'data');
if ~exist(dataDir,'dir'), mkdir(dataDir), end

% Make meg dir
mriDir = fullfile(dataDir,'meg');
if ~exist(mriDir,'dir'), mkdir(mriDir), end

% Make ImageNet dir
trainDir = fullfile(mriDir,'imagenet');
if ~exist(trainDir,'dir'), mkdir(trainDir),end

% Make subject dir
subDir = fullfile(trainDir,sprintf('sub%02d', subID));
if ~exist(subDir,'dir'), mkdir(subDir),end

% Make session dir
sessDir = fullfile(subDir,sprintf('sess%02d', sessID));
if ~exist(sessDir,'dir'), mkdir(sessDir), end

%% Screen setting
Screen('Preference', 'SkipSyncTests', 1);
if runID > 1
    Screen('Preference','VisualDebugLevel',3);
end
bkgColor = [0.485, 0.456, 0.406] * 255; % ImageNet mean intensity
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;

%% IO setting
ioObj = io64;
status = io64(ioObj);
address = hex2dec('D020');
if status,error('The driver installation process was not successful'); end 
startMark = 1; endMark = 8; % Mark for begin and end of the recording
stimMark = 2; respMark = 4; % Mark for stimulus onset and response timing
markDur = 0.005;

%% Key setting
KbName('UnifyKeyNames'); 
startKey = KbName('s');
escKey = KbName('ESCAPE');
% Left hand for animate and right hand for inanimate
animateKey1 = KbName('1!'); % Left hand:1!
animateKey2 = KbName('2@'); % Left hand:2@
inanimateKey1 = KbName('3#'); % Right hand: 3#
inanimateKey2 = KbName('4$'); % Right hand: 4$

%% Make design for this session
% Set design dir
designDir = fullfile(workDir,'stimulus','imagenet','designMatrix');
designFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%02d_design.mat',subID,sessID));
if ~exist(designFile,'file')
    load(fullfile(designDir,'BIN.mat'),'BIN');
    if subID <= 10
        sess = 4*(subID-1)+ sessID;
    else
        sess = 40 + 2*(subID-11) + sessID;
    end
    
    % For each session, we have 5 runs, 200 images/run
    classID = randperm(1000);
    stimulus= reshape(BIN.stimulus(classID,sess),[200,nRun]);
    className = reshape(BIN.classID(classID), [200,nRun]);
    superClassName = reshape(BIN.superClassName(classID), [200,nRun]);
    superClassID  = reshape(BIN.superClassID(classID), [200,nRun]);
    save(designFile,'stimulus','classID','className',...
        'superClassID','superClassName');
end

% Load session design
load(designFile,'stimulus','className', 'classID');

% Image for this run
runStim = stimulus(:,runID); % 200 x 5 cell array 
runClassName = className(:,runID); % 200 x 5 cell array

% Collect trial info for this run
% [class, onset, dur, soa, key, rt]
nStim = length(runStim);
nTrial = nStim;
trial = zeros(nTrial, 6); % [class, onset, dur, soa, key, rt]
classID = reshape(classID, [200,nRun]);
trial(:,1) = classID(:,runID); 
jit = [1.3, 1.7]; % random trial length 
% soa = jit(1) + (jit(2)-jit(1)) * rand(nTrial,1);
jitter = rand(nTrial,1); % soa, [1.3£¬1.7]
jitter = jitter - sum(jitter)/nTrial;
soa = rescale(jitter, jit(1), jit(2));
trial(:,4) = soa; 

%% Load stimulus and instruction
% Visule angle for stimlus and fixation
imgAngle = 16; fixOuterAngle = 0.2; fixInnerAngle = 0.1;

% Visual angle to pixel
% pixelPerMilimeterHor = 1024/390;
% pixelPerMilimeterVer = 768/295;
pixelPerMilimeterHor = 1024/419;
pixelPerMilimeterVer = 768/315;
imgPixelHor = round(pixelPerMilimeterHor * (2 * 1000 * tan(imgAngle/180*pi/2)));
imgPixelVer = round(pixelPerMilimeterVer * (2 * 1000 * tan(imgAngle/180*pi/2)));
fixOuterSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixOuterAngle/180*pi/2)));
fixInnerSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixInnerAngle/180*pi/2)));

% Load stimuli
stimDir = fullfile(workDir,'stimulus','imagenet','images');
img = cell(nStim,1);
for t = 1:nStim
    imgFile = fullfile(stimDir, runClassName{t}, runStim{t});
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

fprintf(['*** Please ask MEG operator to turn on MEG.\n' ...
    '*** Afte MEG has been turn on, press S key to begin the exp.\n'])

% Set trigger(S key) to begin the experiment
while KbCheck(); end
while true
    [keyIsDown,tKey,keyCode] = KbCheck();
    if keyIsDown && keyCode(startKey)
        % Mark begining of exp 
        io64(ioObj,address,startMark);
        while GetSecs - tKey < markDur; end
        io64(ioObj,address,0);
        break;
    elseif keyIsDown && keyCode(escKey)
        sca; return;
    end
end

%% Run experiment
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 0.7 - 0.5*flipInterval; % on duration for a stimulus
beginDur = 1; % beigining fixation duration
endDur = 1; % ending fixation duration
fixOuterColor = [0 0 0]; % color of fixation circular ring
fixInnerColor = [255 255 255]; % color of fixation circular point

% Show begining fixation
% Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
% Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor ,[], 2);
Screen('DrawDots', wptr, [xCenter,xCenter;yCenter,yCenter], ...
    [fixOuterSize, fixInnerSize], [fixOuterColor', fixInnerColor'], [], 2);
Screen('DrawingFinished',wptr);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
% sti(0.7) --> fix( 0.6-1.0) --> next trial
tStart = GetSecs;
for t = 1:nTrial
    % Show stimulus with fixation
    stimTexture = Screen('MakeTexture', wptr, img{t});
    Screen('PreloadTextures',wptr,stimTexture);
    Screen('DrawTexture', wptr, stimTexture); Screen('Close',stimTexture);
%     Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
%     Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,xCenter;yCenter,yCenter], ...
        [fixOuterSize, fixInnerSize], [fixOuterColor', fixInnerColor'], [], 2);
    Screen('DrawingFinished',wptr);
    tStim = Screen('Flip',wptr);
    % Mark onset of the stimulus
    io64(ioObj,address,stimMark);
    while GetSecs - tStim < markDur, end
    io64(ioObj,address,0);
    trial(t, 2) = tStim - tStart; % stimulus onset
    
    % If subject responds in stimulus presenting, we record it
    key = 0; rt = 0;
    while KbCheck(), end % empty the key buffer
    while GetSecs - tStim < onDur
        [keyIsDown, tKey, keyCode] = KbCheck();
        if keyIsDown
            % Mark the rsponese
            io64(ioObj,address,respMark);
            while GetSecs - tKey < markDur, end
            io64(ioObj,address,0);
            if keyCode(escKey),sca; return;
            elseif keyCode(animateKey1) || keyCode(animateKey2)
                key = 1; rt = tKey - tStim;
            elseif keyCode(inanimateKey1)|| keyCode(inanimateKey2)
                key = -1; rt = tKey - tStim;
            end
        end
    end
  
    % Show fixation
%     Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
%     Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor ,[], 2);
    Screen('DrawDots', wptr, [xCenter,xCenter;yCenter,yCenter], ...
        [fixOuterSize, fixInnerSize], [fixOuterColor', fixInnerColor'], [], 2);
    Screen('DrawingFinished',wptr);
    tFix = Screen('Flip', wptr);
    trial(t, 3) = tFix - tStim; % stimulus duration

    % If subject has ready responded in stimtulus presenting, we'll not
    % record it in fixation period; if not, we record it.
    if rt
        while GetSecs - tStim < soa(t)
            [keyIsDown, ~, keyCode] = KbCheck();
            if keyIsDown && keyCode(escKey), sca; return; end
        end
    else
        while GetSecs - tStim < soa(t)
            [keyIsDown, tKey, keyCode] = KbCheck();
            if keyIsDown
                % Mark the response
                io64(ioObj,address,respMark);
                while GetSecs - tKey < markDur, end
                io64(ioObj,address,0);
                if keyCode(escKey),sca; return;
                elseif keyCode(animateKey1) || keyCode(animateKey2)
                    key = 1; rt = tKey - tStim;
                elseif keyCode(inanimateKey1)|| keyCode(inanimateKey2)
                    key = -1; rt = tKey - tStim;
                end
            end
        end
    end
    trial(t, 5:6) = [key,rt];
end

% Mark ending of exp 
tEnd = GetSecs;
io64(ioObj,address,endMark);
while GetSecs - tEnd < markDur, end
io64(ioObj,address,0);

% Show end instruction
endTexture = Screen('MakeTexture', wptr, imgEnd);
Screen('PreloadTextures',wptr,endTexture);
Screen('DrawTexture', wptr, endTexture);
Screen('DrawingFinished',wptr);
Screen('Flip', wptr);
WaitSecs(endDur)

% Show cursor and close all
ShowCursor;
Screen('CloseAll');

%% Evaluate the response
load(fullfile(designDir,'animate_or_not.mat'),'animate_label');
% trial, nTial * 6 array;  % [class, onset, dur, isi, key, RT]
% Make target matrix nTrial x nCond
target = zeros(nTrial,2);
animate_label = animate_label(trial(:,1));
target(:,1) = animate_label == 1;
target(:,2) = animate_label == -1;

% Make response matrix nTrial x nCond
response = zeros(nTrial,2);
response(:,1) = trial(:,5) == 1;
response(:,2) = trial(:,5) == -1;

% Summarize the response with figure 
responseEvaluation(target, response,{'Animate', 'Inanimate'});

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
fprintf('MEG ImageNet:sub%d-sess%d-run%d ---- DONE!\n',...
    subID, sessID,runID)

function responseEvaluation(target,response,condName)
% responseEvaluation(target,response,condName)
% target, response,rt,condName
idx = any(response,2);% only keep trial with response
[cVal,cMat,~,cPer] = confusion(target(idx,:)',response(idx,:)');
figure('Units','normalized','Position',[0 0 0.5 0.5])
% subplot(1,2,1), 
imagesc(cMat);
title(sprintf('RespProp = %.2f, Accuracy = %.2f',sum(idx)/length(target) ,1-cVal));
axis square
set(gca,'Xtick',1:length(cMat), 'XTickLabel',condName,...
    'Ytick',1:length(cMat),'YTickLabel',condName);
colorbar
text(0.75,1,sprintf('%.2f',cPer(1,3)),'FontSize',50,'Color','r');% hit
text(0.75,2,sprintf('%.2f',cPer(1,1)),'FontSize',50,'Color','r');% miss
text(1.75,1,sprintf('%.2f',cPer(1,2)),'FontSize',50,'Color','r');% false alarm
text(1.75,2,sprintf('%.2f',cPer(1,4)),'FontSize',50,'Color','r');% corect reject

% subplot(1,2,2), bar(cPer);
% set(gca,'XTickLabel',condName);
% ylabel('Rate')
% axis square
% legend({'Miss','False alarm','Hit','Correct reject'},...
%    'Orientation','vertical' ,'Location','northeastoutside' )
% legend boxoff






