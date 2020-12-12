function objectRestingMRI(subID, sessID, runID, runDur)
% function objectRestingMRI(subID, sessID, runID, runDur)
% fMRI experiment for BrainImageNet test dataset
% subID, subjet ID, integer[1-20]
% runID, run ID, integer [1-10]
% workdir(or codeDir) -> sitmulus/instruciton/data 
if nargin < 4, runDur = 480; end
if nargin < 3, runID = 1; end 

%% Check subject information
% Check subject id
if ~ismember(subID, 1:20), error('subID is a integer within [1:20]!'); end
% Check session id
if ~ismember(sessID, 1:5), error('sessID is a integer within [1:5]!');end
% Check run id, max 2 runs in a session
if ~ismember(runID, 1:2), error('runID is a integer within [1:2]!'); end

%% Data dir
workDir = pwd;


%% Screen setting
Screen('Preference', 'SkipSyncTests', 1);
Screen('Preference','VisualDebugLevel',4);
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

%% Load stimulus and instruction
fixOuterAngle = 0.3;
fixInnerAngle = 0.2;
% Visual angle to pixel
pixelPerMilimeterHor = 1024/390;
fixOuterSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixOuterAngle/180*pi/2)));
fixInnerSize = round(pixelPerMilimeterHor * (2 * 1000 * tan(fixInnerAngle/180*pi/2)));

% Load instruction image
imgStart = imread(fullfile(workDir, 'instruction', 'restStart.JPG'));
imgEnd = imread(fullfile(workDir, 'instruction', 'restEnd.JPG'));

%% Show instruction
startTexture = Screen('MakeTexture', wptr, imgStart);
Screen('DrawTexture', wptr, startTexture);
Screen('Flip', wptr);
Screen('Close',startTexture);

% Wait ready signal from subject
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && (keyCode(cueKey1) || keyCode(cueKey2)), break; end
end
redFixation = [255 0 0]; % read fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixOuterSize, redFixation, [], 2);
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
% Show begining fixation
whiteFixation = [255 255 255]; % white fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
tStart = Screen('Flip',wptr);

while GetSecs - tStart < runDur
    [keyIsDown, ~, keyCode] = KbCheck();
    if keyIsDown && keyCode(escKey),sca; return;  end
end

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

% Print  info
fprintf('BIN Resting fMRI:sub%d-sess%d-run%d ---- DONE!\n',...
    subID, sessID,runID)















