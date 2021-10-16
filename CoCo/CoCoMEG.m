function trial = CoCoMEG(subID, sessID, runID)
% function trial = CoCoMEG(subID, sessID, runID)
% fMRI experiment for BrainImageNet test dataset
% subID, subjet ID, integer[1-20]
% runID, run ID, integer [1-10]
% workdir(or codeDir) -> sitmulus/instruciton/data

%% Check subject information
% Check subject id
if ~ismember(subID, 1:20), error('subID is a integer within [1:10]!'); end
% Check session id
if ~ismember(sessID, 1:1), error('sessID is a integer within [1:1]!');end
% Check run id
nRun = 10;
if ~ismember(runID, 1:nRun), error('runID is a integer within [1:10]!'); end

%% Data dir
% Make work dir for MEG COCO
workDir = pwd;
% Make data dir
dataDir = fullfile(workDir,'data');
if ~exist(dataDir,'dir'), mkdir(dataDir), end
% Make meg dir
mriDir = fullfile(dataDir,'meg');
if ~exist(mriDir,'dir'), mkdir(mriDir), end
% Make coco dir for the subject
testDir = fullfile(mriDir,'coco');
if ~exist(testDir,'dir'), mkdir(testDir),end
% Make subject dir
subDir = fullfile(testDir,sprintf('sub%02d', subID));
if ~exist(subDir,'dir'), mkdir(subDir),end
% Make session dir
sessDir = fullfile(subDir,sprintf('sess%02d', sessID));
if ~exist(sessDir,'dir'), mkdir(sessDir), end

%% Screen setting
Screen('Preference', 'SkipSyncTests', 1);
if runID > 1
    Screen('Preference','VisualDebugLevel',3);
end
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
bkgColor = [0.485, 0.456, 0.406] * 255; % ImageNet mean intensity
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;

%% IO setting
ioObj = io64;
status = io64(ioObj);
address = hex2dec('D020');
if status,error('The driver installation process was successful'); end
startMark = 1; endMark = 8; % Mark for begin and end of the recording
stimMark = 2; respMark = 4; % Mark for stimulus onset and response timing
markDur = 0.005;

%% keys setting
% PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
startKey = KbName('s');
escKey = KbName('ESCAPE');
% Left hand for animate and right hand for inanimate
animateKey1 = KbName('8*'); % Left hand:8*
animateKey2 = KbName('2@'); % Left hand:2@
inanimateKey1 = KbName('6^'); % Right hand: 6^
inanimateKey2 = KbName('4$'); % Right hand: 4$

%% Load stimulus and instruction
imgAngle = 16;
fixOuterAngle = 0.2;% 0.3
fixInnerAngle = 0.1;% 0.2
% Visual angle to pixel
pixelPerMilimeterHor = 1024/419;
pixelPerMilimeterVer = 768/315;
imgPixelHor = round(pixelPerMilimeterHor * (2 * 751 * tan(imgAngle/180*pi/2)));
imgPixelVer = round(pixelPerMilimeterVer * (2 * 751 * tan(imgAngle/180*pi/2)));
fixOuterSize = round(pixelPerMilimeterHor * (2 * 751 * tan(fixOuterAngle/180*pi/2)));
fixInnerSize = round(pixelPerMilimeterHor * (2 * 751 * tan(fixInnerAngle/180*pi/2)));

% Load stimulus
stimDir = fullfile(workDir,'stimulus','coco','images');
imgName = extractfield(dir(stimDir), 'name');
imgName = imgName(3:end);
nStim = length(imgName);
img = cell(nStim,1);
for t = 1:nStim
    img{t}  = imresize(imread(fullfile(stimDir, imgName{t})), [imgPixelHor imgPixelVer]);
end

% Load instruction image
imgStart = imread(fullfile(workDir, 'instruction', 'trainStart.JPG'));
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
%     if keyIsDown && (keyCode(cueKey1) || keyCode(cueKey2)), break;  end
    if keyIsDown && (keyCode(animateKey1) || keyCode(animateKey2)), break; 
    end
end
readyDotColor = [255 0 0];
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, readyDotColor, [], 2);
Screen('Flip', wptr);

fprintf(['*** Please ask MEG console to turn on MEG.\n' ...
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

%% Make design
% [img, onset, dur, isi, key, rt]
nTrial = 240;
trial = zeros(nTrial, 6); % [imgid, onset, dur, soa, key, rt]

% Each of 120 images will be repeated twice
imgid = [1:120, 1:120];%  image imgid
while true
    imgid = imgid(randperm(length(imgid)));
    if all(diff(imgid)),break; end
end
trial(:,1) = imgid;

% Make random soa with mean 1.5 sec
trange = [1.3, 1.7]; % random trial duration 
soa = trange(1) + (trange(2)-trange(1)) * rand(nTrial,1);
trial(:,4) = soa - (mean(soa) - mean(trange)); % shift soa with mean as mean of trange

%% Run experiment
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 0.7 - 0.5*flipInterval; % on duration for a stimulus
beginDur = 5; % beigining fixation duration
endDur = 5; % ending fixation duration
fixColor = [0 0 0; 255 255 255]'; % color of fixation 
fixCenter = [xCenter,yCenter; xCenter,yCenter]';
fixSize =  [fixOuterSize, fixInnerSize];

% Show begining fixation
Screen('DrawDots', wptr, fixCenter, fixSize, fixColor, [], 2);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
% sti(0.7) --> fix( 0.6-1.0) --> next trial
tStart = GetSecs;
for t = 1:nTrial
    stimTexture = Screen('MakeTexture', wptr, img{trial(t,1)});
    Screen('DrawTexture', wptr, stimTexture); Screen('Close',stimTexture);
    Screen('DrawDots', wptr, fixCenter, fixSize, fixColor, [], 2);
    Screen('DrawingFinished',wptr);
    tStim = Screen('Flip',wptr);
    % Mark onset of the stimulus
    io64(ioObj,address,stimMark);
    while GetSecs - tStim < markDur, end
    io64(ioObj,address,0);
    trial(t, 2) = tStim - tStart; % stimulus onset
    
    % If subject respond in stimulus presenting, we record it
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
    
    % Show after stimulus fixation
    Screen('DrawDots', wptr, fixCenter, fixSize, fixColor, [], 2);
    Screen('DrawingFinished',wptr);
    tFix = Screen('Flip',wptr);
    trial(t, 3) = tFix - tStim; % stimulus duration
    
    
    % If subject have ready responded in stimtulus presenting, we'll not
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
    end
    trial(t, 5:6) = [key, rt];
end

% Mark ending of exp
tEnd = GetSecs;
io64(ioObj,address,endMark);
while GetSecs - tEnd < markDur, end
io64(ioObj,address,0);

% Show end instruction
endTexture = Screen('MakeTexture', wptr, imgEnd);
Screen('PreloadTextures',wptr,endTexture);
Screen('DrawTexture', wptr, endTexture);Screen('Close',endTexture);
Screen('DrawingFinished',wptr);
Screen('Flip', wptr);
WaitSecs(endDur);

% Show cursor and close all
ShowCursor;
Screen('CloseAll');

%% Evaluate the response
% load(fullfile(designDir,'coco_animate_or_not.mat'),'animate_label');
cocoDir = fullfile(workDir, 'stimulus', 'coco');
load(fullfile(cocoDir,'coco_animate_or_not.mat'),'animate_label');
% trial, nTial * 6 array;  % [class, onset, dur, soa, key, RT]
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
fprintf('MEG COCO:sub%d-sess%d-run%d ---- DONE!\n',...
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














