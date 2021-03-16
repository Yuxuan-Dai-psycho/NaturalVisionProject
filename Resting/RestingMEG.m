function RestingMEG(subID, sessID, runID, runDur)
% function RestingMRI(subID, sessID, runID, runDur)
% Resting MEG scan
% subID, subjet ID, integer[1-20]
% runID, run ID, integer [1-10]
% workdir(or codeDir) -> sitmulus/instruciton/data 
if nargin < 4, runDur = 480; end
if nargin < 3, runID = 1; end 

%% Check subject information
% Check subject id
if ~ismember(subID, [1:20 10086]), error('subID is a integer within [1:20]!'); end
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
cueKey1 = KbName('1!');
cueKey2 = KbName('2@');

%% IO setting
ioObj = io64;
status = io64(ioObj);
address = hex2dec('D020');
if status,error('The driver installation process was successful'); end 
startMark = 1; endMark = 8; % Mark for begin and end of the recording
% stimMark = 2; respMark = 4; % Mark for stimulus onset and response timing
markDur = 0.005;

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

%% Run experiment
% Show begining fixation
whiteFixation = [255 255 255]; % white fixation
Screen('DrawDots', wptr, [xCenter,yCenter], fixInnerSize, whiteFixation , [], 2);
tStart = Screen('Flip',wptr);

while GetSecs - tStart < runDur
    [keyIsDown, ~, keyCode] = KbCheck();
    if keyIsDown && keyCode(escKey),sca; return;  end
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

% Print  info
fprintf('Resting MEG:sub%d-sess%d-run%d ---- DONE!\n',...
    subID, sessID,runID)















