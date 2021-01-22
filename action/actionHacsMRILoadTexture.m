% function trial = actionHacsMRILoadTexture(subID,sessID,runID)
% function [subject,task] = actionHacsMRI(subID,sessID,runID)
% Action HACS fMRI experiment stimulus procedure
% 30 subject will do one session, the best 10 from them will do another 3 sessions
% Subject do passive view task
% subID, subjet ID, integer[1-30]
% sessID, session ID, integer [1-4]
% runID, run ID, integer [1-8]
% workdir(or codeDir) -> sitmulus/instruciton/data 
% if nargin < 3, sessID = 1; end
subID=1;sessID=1;runID=1;

%% Check subject information
% Check subject id
if ~ismember(subID, [1:30,10086]), error('subID is a integer within [1:30]!'); end
% Check session id
if ~ismember(sessID, 1:4), error('sessID is a integer within [1:4]!');end
% Check run id
if ~ismember(runID, 1:8), error('runID is a integer within [1:8]!'); end
nRun = 8;
nRepeat = nRun/2; % repeat time of classes in one session
% Check continued subject id
continuedSubID = []; % complete it after the first part experiment.
if (sessID > 1) && (~ismember(subID, continuedSubID))
    error(['subID is not in continuedID within ' mat2str(continuedSubID)]); end

%% Data dir
% Make work dir
workDir = 'D:\fMRI\action';

% Make data dir
dataDir = fullfile(workDir,'data');
if ~exist(dataDir,'dir'), mkdir(dataDir), end

% Make fmri dir
mriDir = fullfile(dataDir,'fmri');
if ~exist(mriDir,'dir'), mkdir(mriDir), end

% Make subject dir
subDir = fullfile(mriDir,sprintf('sub%02d', subID));
if ~exist(subDir,'dir'), mkdir(subDir),end

% Make session dir
sessDir = fullfile(subDir,sprintf('sess%02d', sessID));
if ~exist(sessDir,'dir'), mkdir(sessDir), end
%% for Test checking
if subID ==10086
   subID = 1; 
   Test = 1;
else
    Test = 0;
end
%% Screen setting
Screen('Preference', 'SkipSyncTests', 2);
if runID > 1
    Screen('Preference','VisualDebugLevel',3);
end
Screen('Preference','VisualDebugLevel',4);
Screen('Preference','SuppressAllWarnings',1);
bkgColor = [128 128 128]; % For no specific reason, set median of 255 
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
HideCursor;

% Visule angle for stimlus and fixation
videoAngle = 16;
fixAngle = 0.5;

% Visual angle to pixel
pixelPerMilimeterHor = 1024/390;
pixelPerMilimeterVer = 768/295;
videoPixelHor = round(pixelPerMilimeterHor * (2 * 1000 * tan(videoAngle/180*pi/2)));
videoPixelVer = round(pixelPerMilimeterVer * (2 * 1000 * tan(videoAngle/180*pi/2)));
fixSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixAngle/180*pi/2)));

% define size rect of the video frame
dsRect = [xCenter-videoPixelHor/2, yCenter-videoPixelHor/2,...
    xCenter+videoPixelVer/2, yCenter+videoPixelVer/2];

%% Response keys setting
% PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
startKey = KbName('s');
escKey = KbName('ESCAPE');

% Left hand for interior and right hand for exterior
animateKey1 = KbName('1!'); % Left hand:1!
animateKey2 = KbName('2@'); % Left hand:2@
inanimateKey1 = KbName('3#'); % Right hand: 3#
inanimateKey2 = KbName('4$'); % Right hand: 4$

%% Make design for this session
% Set design dir
designDir = fullfile(workDir,'stimulus','designMatrix');
designFile = fullfile(sessDir,...
    sprintf('sub%02d_sess%02d_design.mat',subID,sessID));
if ~exist(designFile,'file')
    load(fullfile(designDir,'action.mat'),'action');
    if sessID == 1, sess = subID; % For the first part experiment.
    else, sess = 30 + 3*(find(continuedSubID==subID)-1) + sessID-1; end
    % prepare stimulus order and onset info
    sessPar = squeeze(action.paradigmClass(:,sess,:));
    sessStim = squeeze(action.stimulus(:,sess));
    sessClass = cell(200, nRepeat);
    classOrder = sessPar(:,2);
    classOrder = reshape(classOrder,[200,nRepeat]);
    sessStim = reshape(sessStim,[200,nRepeat]);
    for r = 1:nRepeat % random stim order for each 200 classes
        sessStim(:,r) = sessStim(classOrder(:,r), r);
        sessClass(:,r) = action.className(classOrder(:,r));
    end
    sessStim = reshape(sessStim,[100,nRun]);
    sessClass = reshape(sessClass, [100,nRun]);
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
trial = zeros(nTrial, 4); % [onset, class, dur, timing error]
trial(:,1:3) = squeeze(sessPar(:,runID,:)); % % [onset, class, dur]

%% Load stimulus and instruction
% Load stimuli
stimDir = fullfile(workDir, 'stimulus', 'video');
videoTextureSum = cell(nStim, 1);
count = 1;          % Number of loaded movie frames.
for t = 1:nStim
    videoPath = fullfile(stimDir, runClass{t}, runStim{t});
    [video,videoDur,fps] = Screen('OpenMovie', wptr, videoPath);
    Screen('PlayMovie', video, 100, 0, 0);
    % Initialize a texture container for this clip
    videoTextureTmp = zeros(ceil(videoDur * fps), 1); 
    while 1
        textureHandle = Screen('GetMovieImage', wptr, video);
        if textureHandle <= 0, break; end        % End of movie. break out of loop.
        videoTextureTmp(count) = textureHandle;  % Store its texture handle 
        count = count + 1;                       % Update count frame
    end
    videoTextureSum{t} = videoTextureTmp;
    Screen('CloseMovie', video); % close movie file
end

% Load  instruction
imgStart = imread(fullfile(workDir, 'instruction', 'expStart.JPG'));
imgEnd = imread(fullfile(workDir, 'instruction', 'expEnd.JPG'));

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
Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, readyDotColor, [], 2);
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
onDur = 2 - 0.5*flipInterval; % on duration for a stimulus
runDur = 480; % duration for a run
beginDur = 1; % beigining fixation duration
endDur = 1; % ending fixation duration
fixColor = [255 255 255]; % color of fixation 
fixThickness = 2; % thickness of fixation 
tEnd = [trial(2:end, 1);runDur]; % make sequence of tEnd

% Show begining fixation
Screen('FrameOval', wptr, fixColor, [xCenter-fixSize/2, yCenter-fixSize/2,...
    xCenter+fixSize/2, yCenter+fixSize/2], fixThickness);
Screen('DrawingFinished',wptr);
Screen('Flip',wptr);
WaitSecs(beginDur);

% Show stimulus
tStart = GetSecs;
for t = 1:nTrial
    % Show stimulus with fixation
    videoTexture = videoTextureSum{t};
    count = size(videoTexture, 2);
    currentIndex = 1; % current index of frame
    tStim = GetSecs;
    trial(t, 4) = tStim - tStart; % timing error
    
    % If press escape, then break the experiment
    while KbCheck(), end % empty the key buffer
    while GetSecs - tStim < onDur
        tex = videoTexture(currentIndex);
        Screen('DrawTexture', wptr, tex, [], dsRect);
        % wait response
        [keyIsDown, ~, keyCode] = KbCheck();
        if keyIsDown
            if keyCode(escKey),sca; return; end
        end
        % Draw frame on the screen
        Screen('FrameOval', wptr, fixColor, [xCenter-fixSize/2, yCenter-fixSize/2,...
            xCenter+fixSize/2, yCenter+fixSize/2], fixThickness);
        Screen('DrawingFinished',wptr);
        Screen('Flip', wptr);
        Screen('Close', tex)
        % Update frame index
        currentIndex = mod(currentIndex, count) + 1;
    end
        
    % Show fixation
    Screen('FrameOval', wptr, fixColor, [xCenter-fixSize/2, yCenter-fixSize/2,...
        xCenter+fixSize/2, yCenter+fixSize/2], fixThickness);
    Screen('DrawingFinished',wptr);
    Screen('Flip', wptr);
    
    % If press escape, then break the experiment
    while KbCheck(), end % empty the key buffer
    while GetSecs - tStart < tEnd(t)
        [keyIsDown, ~, keyCode] = KbCheck();
        if keyIsDown
            if keyCode(escKey),sca; return; end
        end
    end
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

%% Save data for this run
clear imgStart imgEnd
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
fprintf('action HACS fMRI:sub%d-sess%d-run%d ---- DONE!\n',...
    subID, sessID,runID)
if Test == 1
    fprintf('Testing action HACS fMRI ---- DONE!\n')
end


