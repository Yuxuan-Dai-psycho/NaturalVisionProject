function ImageNetMemoryPrep(subID,sessID, nTrial)
% function trial = ImageNetMemoryPrep(subID,sessID)
% Memory test training after BrianImageNet fMRI train experiment
% subID, subjet ID, integer[1-20]
% sessID, session ID, integer [1-4]
% ntrial, trials of preparing behavior, integer[1:1000]
% workdir(or codeDir) -> sitmulus/instruciton/data

if nargin < 3, nTrial = 20; end

%% Check subject information
% Check subject id
if ~ismember(subID, [1:20 10086]), error('subID is a integer within [1:20]!'); end
% Check session id
if ~ismember(sessID, 1:4), error('sessID is a integer within [1:4]!');end
% Check session id
if ~ismember(sessID, 1:1000), error('sessID is a integer within [1:1000]!');end

%% Data dir
% All dir should exist before the behavior test
workDir = pwd;

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
% 1000 category image from next two subjects are loaded for preparing behavior
% BIN.classID is ImageNet class id, 1000x1, cell array
% BIN.stimulus is stimlus filename, 1000 x 80, cell array
designDir = fullfile(workDir,'stimulus','imagenet','designMatrix');
load(fullfile(designDir,'BIN.mat'),'BIN');
% session id of this subject and next two subejct
sess = 4*(subID+1) + sessID;
sess =  mod(sess,80);
% 1000 category and each has an example
categoryName = repmat(BIN.classID,length(sess),1);
exampleName = reshape(BIN.stimulus(:,sess),[],1);
nStim = length(exampleName);

% Randomize stimulus
idx = randperm(nStim);
categoryName = categoryName(idx);
exampleName = exampleName(idx);

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
onDur = 2.5 - 0.5*flipInterval; % on duration for a stimulus
maskDur = 0.2; % ending duration of each trial
maxDur = 2; % max duration of a trial
beginDur = 4; % beigining fixation duration
endDur = 4; % ending fixation duration
fixOuterColor = [0 0 0]; % color of fixation circular ring
fixInnerColor = [255 255 255]; % color of fixation circular point
stimDir = fullfile(workDir,'stimulus','imagenet','images');

% Load instruciton and stimuli
imgStart = imread(fullfile(workDir, 'instruction', 'behStart.JPG'));
imgEnd = imread(fullfile(workDir, 'instruction', 'behEnd.JPG'));
img = cell(nTrial,1);
for t = 1:nTrial
    imgFile = fullfile(stimDir, categoryName{t}, exampleName{t});
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

% Show stimulus for each trial
for t = 1:nTrial
    % Show stimulus with fixation
    stimTexture = Screen('MakeTexture', wptr, img{t});
    Screen('PreloadTextures',wptr,stimTexture);
    Screen('DrawTexture', wptr, stimTexture);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor, [], 2);
    Screen('DrawingFinished',wptr);
    tStim = Screen('Flip',wptr);
    

    % Record response while stimulus is on
    rt = 0;
    while KbCheck(), end % empty the key buffer
    while GetSecs - tStim < onDur
        if GetSecs - tStim > 1.5
            Screen('DrawTexture', wptr, stimTexture);
            Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
            Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, [255 0 0], [], 2);
            Screen('DrawingFinished',wptr);
            Screen('Flip',wptr);
        end
        [keyIsDown, tKey, keyCode] = KbCheck();
        if keyIsDown
            if keyCode(escKey),sca; return;
            elseif keyCode(seenKey)
                rt = tKey - tStim; break;
            elseif keyCode(notSeenKey)
                rt = tKey - tStim;break;
            end
        end
    end
    Screen('Close',stimTexture);
    % Show fixation
    Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, fixOuterColor, [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, fixInnerColor , [], 2);
    Screen('DrawingFinished',wptr);
    tFix = Screen('Flip', wptr);

    if rt % If subject already responds,just show fixation with short time
        while GetSecs - tFix < maskDur, end
    else % if subejct have not responded, wait the response until the end of the trial
        while GetSecs - tStim < maxDur
            [keyIsDown, ~, keyCode] = KbCheck();
            if keyIsDown
                if keyCode(escKey),sca; return;
                    elseif keyCode(seenKey), break;
                    elseif keyCode(notSeenKey), break;
                end
            end
        end
    end
end

% Wait ending fixation
endTexture = Screen('MakeTexture', wptr, imgEnd);
Screen('DrawTexture', wptr, endTexture);
Screen('Flip', wptr);
Screen('Close',endTexture);
WaitSecs(endDur);

% Show cursor and close all
ShowCursor;
Screen('CloseAll');



