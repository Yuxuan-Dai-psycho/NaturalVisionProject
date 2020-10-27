% function [subject,task] = binMRI(subID,sessID,runID)
% function [subject,task] = binMRI(subID,sessID,runID)
% Brain ImageNet fMRI experiment stimulus procedure
% subID, subjet ID, integer[1-20] 
% sessID, session ID, integer [1-4]
% runID, run ID, integer [1-10] 

%% Arguments
% if nargin < 3, subID = 1; end
% if nargin < 2, sessID = 1; end
% if nargin < 1, runID = 1; end

%% Check subject information
% Check subject id
if ~ismember(subID, 1:20)
    warning ('subID is a integer within [1:20]!');
end

% Check session id
if ~ismember(sessID, 1:4)
    warning ('sessID is a integer within [1:4]!');
end

% Check run id
if ~ismember(runID, 1:10)
    warning ('runID is a integer within [1:10]!');
end

%% Dir setting
workDir =  '/nfs/e1/BrainImageNet';
stimDir = fullfile(workDir,'images');
designDir = fullfile(workDir,'designMatrix');
dataDir = fullfile(workDir,'data');

%% Response keys setting
PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
startKey = KbName('s');
escKey = KbName('ESCAPE');
% likeKey = KbName('1!'); % Left hand:1!,2@
% dislikeKey = KbName('3$'); % Right hand: 3#,4$ 
likeKey = KbName('f');
disLikeKey = KbName('j');

%% Screen setting
Screen('Preference', 'SkipSyncTests', 2);
Screen('Preference','VisualDebugLevel',4);
Screen('Preference','SuppressAllWarnings',1);
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
bkgColor = [128 128 128];
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;

%% Create instruction texture
% Makes instruction texture
imgStart = sprintf('%s/%s', picsFolderName, 'Instruction_Start.jpg');
imgRest = sprintf('%s/%s', picsFolderName, 'Instruction_Rest.jpg');
imgEnd = sprintf('%s/%s', picsFolderName, 'Instruction_Bye.jpg');

startTexture = Screen('MakeTexture', wptr, imread(imgStart));
restTexture = Screen('MakeTexture', wptr, imread(imgRest));
endTexture = Screen('MakeTexture', wptr, imread(imgEnd));

%% Show start instruction
Screen('DrawTexture', wptr, startTexture);
Screen('Flip', wptr);
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(startKey), break;
    end
end

%% Load design matrix
load(fullfile(workDir,'BIN.mat'));
sess = 4*(subID-1)+ sessID;
% par for this run
sessPar = reshape(sequeeze(BIN.paradigmClass(:,sess,:)),[100,10,3]);
sessStim = reshape(BIN.stimulus(:,sess),[100,10]);
runStim = sessStim(:,runID);
% Image for this run
imgDir = reshape(BIN.classID, [100,10]);
runImgDir = imgDir(:,runID);

% Collect trial info for this run 
trial = zeros(nStim, 5); % [onset, class, dur, key, rt]
trial(:,1:3) = squeeze(sessionPar(:,r,:)); % % [onset, class, dur]

%% Make stimuli texture
nStim = size(runStim, 1);
stimTexture = zeros(nStim,1);
imgPixel = 800;
for t = 1:nStim
    imgFile = fullfile(stimDir, runImgDir, runStim{t});
    img = imread(imgFile);
    img = imresize(img, [imgPixel imgPixel]);
    stimTexture(t) = Screen('MakeTexture', wptr, img);
end


%% Show instruction
Screen('DrawTexture', wPtr, insTexture);
Screen('Flip', wPtr);
% Wait trigger(S key) to begin the test
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(startKey), break;
    elseif keyIsDown && keyCode(escKey), sca; return;
    end
end

%% Run experiment
onDur = 2; % on duration for a stimulus
offDur = 2; % off duration for a stimulus
runDur = 500; % duration for a run
beginDur = 6; % beigining fixation duration
endDur = 6; % ending fixation duration
fixSize = 10;
fixColor = [255 255 255];

% Show begining fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
tFix = Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
for t = 1:nStim
    % Show stimulus with fixation
    Screen('DrawTexture', wptr, stimTexture(t));
    Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
    tStim = Screen('Flip',wptr);
    while GetSecs - tStim < onDur, end
    
    % Show begining fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
    tFix = Screen('Flip',wptr);
    
    while KbCheck(), end % empty the key buffer
    while GetSecs - tFix < offDur
        [keyIsDown, tKey, keyCode] = KbCheck();       
        if keyIsDown
            if keyCode(escKey), sca; return;
            elseif keyCode(redKey),   key = 1;
            elseif keyCode(greenKey), key = 2;
            elseif keyCode(blueKey),  key = 3;
            else, key = 4;
            end
            rt = tKey - tFix; % reaction time
            trial(t, 4:5) = [key,rt];
            break;
        end
    end
    
    % Wait for the end of this trial
    if t < nStim
        tEnd = trial(t+1,1);   
    else
        tEnd = runDur;
    end  
    while GetSecs - tStim < tEnd - tStim, end
end


% Wait ending fixation
WaitSecs(endDur);

% Show end instruction
Screen('DrawTexture', wptr, endTexture);
Screen('Flip', wptr);
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(insKey), break;
    end
end

% show cursor and close all
ShowCursor;
Screen('CloseAll');

%% Save data for this run
subID = 1;sessID = 1;
subDir = fullfile(dataDir,sprintf('sub%02d', subID));
if ~exist(subDir,'dir')
    mkdir(subDir)
end
sessDir = fullfile(subDir,sprintf('sess%02d', sessID));
if ~exist(sessDir,'dir')
    mkdir(sessDir)
end
fileName = fullfile(sessDir, ...
    sprintf('sub%02d_sess%02d_run%2d.mat',subID,sessID, runID));
fprintf('Data were saved to: %s\n',fileName);
% save(filename,'resultPerSession');





