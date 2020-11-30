function trial = binMRItrainBehaviorNew(subID,sessID)
% function trial = binMRItrainBehavior(subID,sessID)
% Memory test after BrianImageNet fMRI train experiment
% subID, subjet ID, integer[1-20]
% sessID, session ID, integer [1-4]


%% Check subject information
% Check subject id
if ~ismember(subID, 1:20), error('subID is a integer within [1:20]!'); end
% Check session id
if ~ismember(sessID, 1:4), error('sessID is a integer within [1:4]!');end

%% Data dir
workDir = 'D:\fMRI\BrainImageNet\stim';
stimDir = fullfile(workDir,'images');
designDir = fullfile(workDir,'designMatrix');

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


%% Prepare params
imgAngle = 12;
fixOuterAngle = 0.3;
fixInnerAngle = 0.2;
% readyDotColor = [255 0 0];
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
escKey = KbName('ESCAPE');
notSeenKey = KbName('f'); % F key for left hand
seenKey = KbName('j'); % J key for right hand

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
imgStart = sprintf('%s/%s', 'instruction', 'instructionStartTrain.jpg');
imgEnd = sprintf('%s/%s', 'instruction', 'instructionBye.jpg');
startTexture = Screen('MakeTexture', wptr, imread(imgStart));
endTexture = Screen('MakeTexture', wptr, imread(imgEnd));

%% Stimulus for this session
% 1000 category image from next subjects are loaded as control
% 2000 images are randomized and shown in four run(each with 500 images) 

load(fullfile(designDir,'BIN.mat'),'BIN');
% BIN.classID  % ImageNet class id, 1000x1, cell array
% BIN.stimulus  % 1000 x 80, cell array

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
exampleName = exampleName(idx);
cond = cond(idx);
jitter = unifrnd(1,1.5,nStim,1);

% Split stimuli into nruns
nruns = 10;
categoryName = reshape(categoryName, [],nruns);
categoryID = reshape(categoryID, [],nruns);
exampleName = reshape(exampleName, [],nruns);
cond = reshape(cond, [],nruns);
jitter = reshape(jitter, [],nruns);

%% Run experiment
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 0.5 - 0.5*flipInterval; % on duration for a stimulus
beginDur = 2; % beigining fixation duration
endDur = 0.5; % ending duration of each trial
fixOuterColor = [0 0 0]; % color of fixation circular ring
fixInnerColor = [255 255 255]; % color of fixation circular point
nTrial = nStim/nruns;

for runID = 1:nruns
    % Make stimuli texture for this run
%     stimTexture = zeros(nTrial,1);
    for t = 1:nTrial
        imgFile = fullfile(stimDir, categoryName{t,runID}, exampleName{t,runID});
        imgTmp = imread(imgFile);
        img{t} = imresize(imgTmp, [imgPixelHor imgPixelVer]);
    end
    
    % make and preload texture for the first trial 
    stimTexture = Screen('MakeTexture', wptr, img{1}); 
    Screen('PreloadTextures',wptr,stimTexture);
    
    % Show instruction
    Screen('DrawTexture', wptr, startTexture);
    Screen('Flip', wptr);
    % Wait ready signal from subject
    while KbCheck(); end
    while true
        [keyIsDown,~,keyCode] = KbCheck();
        if keyIsDown && keyCode(seenKey), break;
        end
    end
    
    % Show begining fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
    Screen('Flip',wptr);
    WaitSecs(beginDur);
   
    % Collect trial info for this runID
    trial = zeros(nTrial, 6); % [onset, categoryID, cond, key, rt, timing error]
    tEnd = cumsum(onDur + jitter(:,runID)); % end time of trials
    trial(:,1) = [0; tEnd(1:end-1)]; % onset of trials
    trial(:,2:3) = [categoryID(:,runID),cond(:,runID)]; % category id and cond
  
    % Show stimulus
    tStart = GetSecs;
    for t = 1:nTrial
        % load image
%         imgFile = fullfile(stimDir, categoryName{t,runID}, exampleName{t,runID});
%         imgTmp = imread(imgFile);
%         img = imresize(imgTmp, [imgPixelHor imgPixelVer]);
%         stimTexture = Screen('MakeTexture', wptr, img); 
        
        % Show stimulus with fixation
        Screen('DrawTexture', wptr, stimTexture);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
        Screen('DrawingFinished',wptr);
        tStim = Screen('Flip',wptr);
        trial(t, 6) = tStim - tStart; % timing error
        
        % make texture for the next trial
        Screen('Close',stimTexture); % closed the previous img texture
        if t < nTrial
        	stimTexture = Screen('MakeTexture', wptr, img{t+1});
        	Screen('PreloadTextures',wptr,stimTexture);
        end
        
        % Show fixation
        Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
        Screen('DrawingFinished',wptr);
        tFix = Screen('Flip', wptr, tStim + onDur);
        
        while KbCheck(), end % empty the key buffer
        while GetSecs - tStart < tEnd(t)
            [keyIsDown, tKey, keyCode] = KbCheck();
            if keyIsDown
                if keyCode(escKey), sca; return;
                elseif keyCode(seenKey),    key = 1;
                elseif keyCode(notSeenKey), key = -1;
                else,  key = 0;
                end
                rt = tKey - tFix; % reaction time
                trial(t,4:5) = [key,rt];
                break;
            end
        end
        WaitSecs(endDur);
    end
    
    % Wait ending fixation
    Screen('DrawTexture', wptr, endTexture);
    Screen('Flip', wptr);   
    % Wait ready signal from subject
    while KbCheck(); end
    while true
        [keyIsDown,~,keyCode] = KbCheck();
        if keyIsDown && keyCode(seenKey), break;
        end
    end
        
    %% Save data for this runID
    dataFile = fullfile(sessDir,...
        sprintf('sub%02d_sess%02d_run%02d_beh.mat',subID,sessID,runID));
    fprintf('Data were saved to: %s\n',dataFile);
    save(dataFile,'trial','sessID','subID','runID');  
end

% Show cursor and close all
ShowCursor;
Screen('CloseAll');
    