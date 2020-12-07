function trial = binMRItrainBehavior(subID,sessID,sRun)
% function trial = binMRItrainBehavior(subID,sessID)
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
% All dir should exist before the behavior test
workDir = pwd;
trainDir = fullfile(workDir,'data','fmri','train');
sessDir = fullfile(trainDir,sprintf('sub%02d/sess%02d',subID,sessID));

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
escKey = KbName('ESCAPE');
seenKey = KbName('f'); % F key for left hand
notSeenKey = KbName('j'); % J key for right hand

%% Make desgin for this session
% 1000 category image from next subjects are loaded as control
% 2000 images are randomized and shown in four run(each with 500 images)
% BIN.classID is ImageNet class id, 1000x1, cell array
% BIN.stimulus is stimlus filename, 1000 x 80, cell array
designDir = fullfile(workDir,'stimulus','train','designMatrix');
designFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%02d_design_beh.mat',subID,sessID));
if ~exist(designFile,'file')
    load(fullfile(designDir,'BIN.mat'),'BIN');
    % session id of this subject and next subejct
    sess = 4* [subID-1,subID] + sessID;
    sess =  mod(sess,80);
    % 1000 category and each has an example
    categoryName = repmat(BIN.classID,length(sess),1);
    categoryID = repmat((1:1000)',length(sess),1);
    exampleName = reshape(BIN.stimulus(:,sess),[],1);
    nStim = length(exampleName);
    % cond indicate if a picture appears in fMRI test
    cond = [ones(1000,1); zeros(1000,1)];
    
    % Randomize stimulus
    idx = randperm(nStim);
    categoryName = categoryName(idx);
    categoryID  = categoryID(idx);
    exampleName = exampleName(idx);
    cond = cond(idx);
    
    % Split stimuli into N runs
    nRun = 4;
    categoryName = reshape(categoryName,[],nRun);
    categoryID = reshape(categoryID,[],nRun);
    exampleName = reshape(exampleName,[],nRun);
    cond = reshape(cond,[],nRun);
    
    % Save design file
    save(designFile,'nRun','nStim','categoryName','categoryID','exampleName','cond');
end

% Load session design of behavior test
load(designFile,'nRun','nStim','categoryName','categoryID','exampleName','cond');


%% Load stimulus and instruction
imgAngle = 16; fixOuterAngle = 0.3; fixInnerAngle = 0.2;

% Visual angle to pixel
pixelPerMilimeterHor = 1024/390;
pixelPerMilimeterVer = 768/295;
imgPixelHor = round(pixelPerMilimeterHor * (2 * 1000 * tan(imgAngle/180*pi/2)));
imgPixelVer = round(pixelPerMilimeterVer * (2 * 1000 * tan(imgAngle/180*pi/2)));
fixOuterSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixOuterAngle/180*pi/2)));
fixInnerSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixInnerAngle/180*pi/2)));

%% Run experiment
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 1 - 0.5*flipInterval; % on duration for a stimulus
maskDur = 0.2; % ending duration of each trial
maxDur = 2; % max duration of a trial
beginDur = 4; % beigining fixation duration
endDur = 4; % ending fixation duration
fixOuterColor = [0 0 0]; % color of fixation circular ring
fixInnerColor = [255 255 255]; % color of fixation circular point
nTrial = nStim/nRun;
stimDir = fullfile(workDir,'stimulus','train','images');
for runID = sRun:nRun
    % Load instruciton and stimuli
    imgStart = imread(fullfile(workDir, 'instruction', 'behStart.JPG'));
    imgEnd = imread(fullfile(workDir, 'instruction', 'behEnd.JPG'));
    img = cell(nTrial,1);
    for t = 1:nTrial
        imgFile = fullfile(stimDir, categoryName{t,runID}, exampleName{t,runID});
        img{t} = imresize(imread(imgFile), [imgPixelHor imgPixelVer]);
    end
    
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
    
    % Make design
    trial = zeros(nTrial, 5); % [onset, categoryID, cond, key, rt]
    trial(:,2:3) = [categoryID(:,runID),cond(:,runID)]; % category id and cond
    
    % Show stimulus for each trial
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
        
        % Show fixation
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
        Screen('DrawingFinished',wptr);
        tFix = Screen('Flip', wptr);
        
        if rt % If subject already responds,just show fixation with short time
            while GetSecs - tFix < maskDur, end
        else % if subejct have not responded, wait the response until the end of the trial
            while GetSecs - tStim < maxDur
                [keyIsDown, tKey, keyCode] = KbCheck();
                if keyIsDown
                    if keyCode(escKey),sca; return;
                    elseif keyCode(seenKey)
                        key = 1; rt = tKey - tStim; break;
                    elseif keyCode(notSeenKey)
                        key = -1; rt = tKey - tStim; break;
                    end
                end
            end
        end
        trial(t, 4:5) = [key,rt];
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
        if empty(oldFile), n = 1;
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
    fprintf('BINbehavior subID:%d, sessID:%d, runID:%d ---- DONE!',subID, sessID,runID)
end

% Show cursor and close all
ShowCursor;
Screen('CloseAll');



