function trial = binMRItest(subID, sessID, runID)
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
if ~ismember(runID, 1:10), error('runID is a integer within [1:10]!'); end

%% Data dir 
workDir = 'H:\NaturalImageData\stimTest';
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
subDir = fullfile(mriDir,sprintf('sub%02d', subID));
if ~exist(subDir,'dir'), mkdir(subDir),end

% Make session dir
sessDir = fullfile(subDir,sprintf('sess%02d', sessID));
if ~exist(sessDir,'dir'), mkdir(sessDir), end


%% Stimulus
designFile = fullfile(testDir,'design.mat');
if ~exist(designFile,'file')
    imgName = extractfield(dir(stimDir), 'name');
    imgName = imgName(3:end);
    % randomize images for this subject 
    subImg = randperm(length(imgName));    
    nRun = 10;
    % each column contain the image IDs for a run
    runImg = reshape(subImg,[],nRun);
    % all runs use the same sequence, two column: [onset, cond]
    runSeq = load(fullfile(workDir,'testSeq.mat'));
    runSeq = runSeq.designMat;
    save(fullfile(testDir,'design.mat'),'imgName','runImg','runSeq');
end

% load design
design = load(designFile);
runImg = design.runImg(:,runID);
runImgName = design.imgName(runImg); 
runSeq = design.runSeq;

%% Display
imgAngle = 12;
fixOuterAngle = 0.3;
fixInnerAngle = 0.2;
readyDotColor = [255 0 0];
bkgColor = [128 128 128];
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
sameKey = KbName('1!'); % Left hand:1!,2@
diffKey = KbName('3#'); % Right hand: 3#,4$ 

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


%% Make stimuli texture
nStim = size(runImg, 1);
stimTexture = zeros(nStim,1);
for t = 1:nStim
    imgFile = fullfile(stimDir, runImgName{t});
    img = imread(imgFile);
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
    if keyIsDown && keyCode(sameKey), break;
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


%% Parms for stimlus presentation
flipInterval = Screen('GetFlipInterval', wptr);% get dur of frame
onDur = 2 - 0.5*flipInterval; % on duration for a stimulus
offDur = 2; % off duration for a stimulus
runDur = 672; % duration for a run
beginDur = 16; % beigining fixation duration
endDur = 16; % ending fixation duration
fixOuterColor = [0 0 0]; % color of fixation circular ring
fixInnerColor = [255 255 255]; % color of fixation circular point

%% Trials
% Collect trial info for this run
nTrial = length(runSeq);
trial = zeros(nTrial, 6); % [onset,cond,imgID, trueAnswer, key, rt]
trial(:,1:2) = runSeq;    % [onset, condition]
for t = 1:nStim
    trial(trial(:,2) == t,3) = runImg(t);
end

% 1-back true answer, 1,same as the previous one.  
trial(2:end,4) = ~diff(trial(:,2));

% Ending timing of trials
tEnd = [trial(2:end, 1);runDur]; 

%% Run experiment
% Show begining fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
tStart = GetSecs;
for t = 1:nStim
    % Show stimulus with fixation
    Screen('DrawTexture', wptr, stimTexture(t));
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
    tStim = Screen('Flip',wptr);

    % Show begining fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
    tFix = Screen('Flip', wptr, tStim + onDur);
    
    while KbCheck(), end % empty the key buffer
    while GetSecs - tFix < offDur
        [keyIsDown, tKey, keyCode] = KbCheck();       
        if keyIsDown
            if keyCode(escKey), sca; return;
            elseif keyCode(sameKey),   key = 1;
            elseif keyCode(diffKey),   key = -1; 
            end
            rt = tKey - tFix; % reaction time
            trial(t, 5) = key;
            trial(t, 6) = rt;
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

% show cursor and close all
ShowCursor;
Screen('CloseAll');

%% Save data for this run
fileName = fullfile(sessDir,sprintf('sub%02d_run%02d.mat',subID, runID));
fprintf('Data were saved to: %s\n',fileName);
save(fileName,'trial');

