function trial = objectCoCoMEG(subID, sessID, runID)
% function trial = objectCoCoMRI(subID, sessID, runID)
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
mriDir = fullfile(dataDir,'meg');
if ~exist(mriDir,'dir'), mkdir(mriDir), end

% Make test dir for the subject
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
stimDir = fullfile(workDir,'stimulus','coco','images');
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
    if keyIsDown && (keyCode(cueKey1) || keyCode(cueKey2)), break;  end
end
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, redFixation, [], 2);
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
soa = 1.2 + 0.3 * rand(nTrial,1); % soa, [1.2,1.5]
trial(:,4) = soa;

%% Run experiment
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 0.5 - 0.5*flipInterval; % on duration for a stimulus
beginDur = 1; % beigining fixation duration
endDur = 1; % ending fixation duration

% Show begining fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
% sti(0.5) --> fix( 0.7-1.0) --> next trial
tStart = GetSecs;
for t = 1:nTrial
    stimTexture = Screen('MakeTexture', wptr, img{trial(t,1)});
    Screen('DrawTexture', wptr, stimTexture);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation ,[], 2);
    Screen('DrawingFinished',wptr);
    Screen('Close',stimTexture);
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
            elseif keyCode(cueKey1) || keyCode(cueKey2)
                key = 1; rt = tKey - tStim;
            end
        end
    end
    
    % Show after stimulus fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
    Screen('DrawingFinished',wptr);
    tFix = Screen('Flip',wptr);
    trial(t, 3) = tFix - tStim; % stimulus duration
    
    
    % If subject have ready responded in stimtulus presenting, we'll not
    % record it in fixation period; if not, we record it.
    if rt
        while GetSecs - tStart < soa(t)
            [keyIsDown, ~, keyCode] = KbCheck();
            if keyIsDown && keyCode(escKey), sca; return; end
        end
    else
        while GetSecs - tStart < soa(t)
            [keyIsDown, tKey, keyCode] = KbCheck();
            if keyIsDown
                % Mark the rsponese
                io64(ioObj,address,respMark);
                while GetSecs - tKey < markDur, end
                io64(ioObj,address,0);
                
                if keyCode(escKey),sca; return;
                elseif keyCode(cueKey1) || keyCode(cueKey2)
                    key = 1; rt = tKey - tStim;
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
Screen('DrawTexture', wptr, endTexture);
Screen('DrawingFinished',wptr);
Screen('Flip', wptr);
Screen('Close',endTexture);
WaitSecs(endDur);

% Show cursor and close all
ShowCursor;
Screen('CloseAll');


%% Evaluate the response
load(fullfile(designDir,'coco_animate_or_not.mat'),'animate_label');
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
[cVal,cMat,~,cPer] = objectConfusion(target(idx,:)',response(idx,:)');
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














