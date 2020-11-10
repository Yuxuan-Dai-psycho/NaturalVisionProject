% Present ImageNet images for each category and get fMRI signal
%

% Version 1.5 2020/10/26
% Department of Psychology, Beijing Normal University 

% clear
clc; clear; close all;

%% Subject registration
% Input subject info and params
subID = str2double(input('Enter subject id: ','s'));
sessID = str2double(input('Enter session id: ','s'));
runID = str2double(input('Enter run id: ','s'));
imgAngle = str2double(input('Enter img angle: ','s'));
fixAngle = str2double(input('Enter fixation angle: ','s'));
% Check input
if ~ismember(subID, 1:20), error('subID is a integer within [1:20]!'); end
if ~ismember(sessID, 1:4), error('sessID is a integer within [1:4]!'); end
if ~ismember(runID, 1:10), error('runID is a integer within [1:10]!'); end

%% Prepare foldernames and params
picsFolderName = 'instruction';
outFolderName = 'out';
stimulusFolder = 'stim/images';
designMatrixMat = 'stim/BIN.mat';
stimTrailON = 2;
nullBlank = 16;
fixColor = [255 255 255];

% compute image pixel
pixelPerMilimeterHor = 1024/390;
pixelPerMilimeterVer = 768/295;
imgPixelHor = pixelPerMilimeterHor * (2 * 1000 * tan(imgAngle/180*pi/2));
imgPixelVer = pixelPerMilimeterVer * (2 * 1000 * tan(imgAngle/180*pi/2));
fixPixelHor = pixelPerMilimeterHor * (2 * 1000 * tan(fixAngle/180*pi/2));

%% Response keys setting 
PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
insKey = KbName('s');
likeKey = KbName('1');
disLikeKey = KbName('3'); 
escKey = KbName('escape'); % stop and exit

%% Screen setting
% Skip synIRF tests
Screen('Preference', 'SkipSyncTests', 2);
Screen('Preference','VisualDebugLevel',4);
Screen('Preference','SuppressAllWarnings',1);
screenNumber = max(Screen('Screens'));% Set the screen to the secondary monitor
bkgColor = [128 128 128];
[wptr, rect] = Screen('OpenWindow', screenNumber, bkgColor);
[xCenter, yCenter] = RectCenter(rect);% the centre coordinate of the wptr in pixels
% close cursor
HideCursor;

%% Create instruction texture
% Makes instruction texture  
imgStart = sprintf('%s/%s', picsFolderName, 'instructionStart.jpg');
imgRest = sprintf('%s/%s', picsFolderName, 'instructionRest.jpg');
imgEnd = sprintf('%s/%s', picsFolderName, 'instructionBye.jpg');
startTexture = Screen('MakeTexture', wptr, imread(imgStart)); 
restTexture = Screen('MakeTexture', wptr, imread(imgRest)); 
endTexture = Screen('MakeTexture', wptr, imread(imgEnd)); 

%% Show start instruction
Screen('DrawTexture', wptr, startTexture);
Screen('Flip', wptr);
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(insKey), break;
    end
end

%% Run experiment: show stimui and wait response for each trial
% Load design matrix
load(designMatrixMat);
paradigmClass = BIN.paradigmClass;
paradigmSuperClass = BIN.paradigmSuperClass;
classID = BIN.classID;
sessionLevel = 4*(subID-1)+sessID;
% get matrix info
paradigmClassReshape = reshape(paradigmClass(:,sessionLevel,:), 100, 10, 3);
paradigmSuperClassReshape = reshape(paradigmSuperClass(:,sessionLevel,:), 100, 10, 3);
stimOnset = cell2mat(paradigmClassReshape(:, runID, 1));
stimOnset = stimOnset - stimOnset(1,1); %substract the first time
stimImgCondition = paradigmClassReshape(:, runID, 2);
stimSuperCondition = paradigmSuperClassReshape(:, runID, 2);
responseArr = cell(100, 7);% seven columns: onset, superCondition, classCondition, imgCondition, actualOnset, response, RT
responseArr(:, 1) = num2cell(stimOnset);
responseArr(:, 2) = num2cell(stimSuperCondition);
responseArr(:, 4) = stimImgCondition;
% Make stimuli texture
stimTexture = zeros(1,100);
for trail = 1:100
    picName = stimImgCondition{trail, 1};
    picClass = regexp(picName, '_', 'split');
    imgPath = sprintf('%s/%s/%s', stimulusFolder, picClass{1}, picName);
    imgResize = imresize(imread(imgPath), [imgPixelHor imgPixelVer]);
    stimTexture(trail) = Screen('MakeTexture', wptr, imgResize);
    responseArr{trail, 3} = find(strcmp(picClass{1}, classID));
end
% show null trials before stimulus
Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor+5, [50 50 50], [], 2);
Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor, fixColor, [], 2);
Screen('Flip', wptr);
WaitSecs(nullBlank);    
  
% start the experiment
tStart = GetSecs;
trail = 1;
while 1
    % show stimulus at the right time
    stimOnsetSingle = stimOnset(trail, 1);
    tCurrent = GetSecs - tStart;
    if floor(tCurrent) == stimOnsetSingle          
        % ON Trail
        Screen('DrawTexture', wptr, stimTexture(trail));
        Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor+5, [50 50 50], [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor, fixColor, [], 2);
        Screen('Flip', wptr);
        WaitSecs(stimTrailON);    
        % OFF Trail & Record response
        Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor+5, [50 50 50], [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor, fixColor, [], 2);
        tBlank = Screen('Flip', wptr);
        % Wait response
        pressTime = 0;
        response = -1;
        RT = 2;
        while KbCheck(), end % empty the key buffer
        while GetSecs - tBlank < 4-stimTrailON
            [keyIsDown, tEnd, keyCode] = KbCheck();
            if keyIsDown
                if pressTime < 1
                    if keyCode(escKey), response = 'break'; break;
                    elseif keyCode(likeKey),   response = 1; RT = tEnd-tBlank;
                    elseif keyCode(disLikeKey), response = 0; RT = tEnd-tBlank;
                    end
                end
                pressTime = pressTime + 1;
            end
        end
        % record response
        responseArr{trail, 5} = tCurrent;
        responseArr{trail, 6} = response;
        responseArr{trail, 7} = RT ;            
        trail = trail+1;% Move on trail
    end
    % using in debugging
    if strcmp(response, 'break')
        break;
    end
    if trail > 100, break, end %break after 100 trails
end

% show null trials after stimulus
if ~strcmp(response, 'break')
    Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor+5, [50 50 50], [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor, fixColor, [], 2);
    Screen('Flip', wptr);
    WaitSecs(nullBlank);
end

%% Save data
outPath = sprintf('%s/sub%02d', outFolderName, subID);
if ~exist(outPath,'dir')
    mkdir(outPath);
end
fprintf('Data were saved to: %s\n',outPath);
outName = sprintf('%s/session%02d_run%02d.mat', outPath, sessID, runID);
save(outName,'responseArr');

%% Show end instruction
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
