% Present ImageNet images for each category and get fMRI signal
%

% Version 1.5 2020/10/26
% Department of Psychology, Beijing Normal University 

% clear
clc; clear; close all;

%% Subject registration
% Set subject id
subject.subID = str2double(input('Enter subject id: ','s'));
 if subject.subID > 20
     error('Please check your subject id!');
 end

 % Set session id
subject.sessionID = str2double(input('Enter session id: ','s'));
 if subject.sessionID > 4
     error('Please check your session id!');
 end

 % Set angle
imgAngle = str2double(input('Enter img angle: ','s'));
fixAngle = str2double(input('Enter fixation angle: ','s'));

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
likeKey = KbName('f');
disLikeKey = KbName('j'); 
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
imgWait = sprintf('%s/%s', picsFolderName, 'instructionWait.jpg');
imgRest = sprintf('%s/%s', picsFolderName, 'instructionRest.jpg');
imgEnd = sprintf('%s/%s', picsFolderName, 'instructionBye.jpg');
startTexture = Screen('MakeTexture', wptr, imread(imgStart)); 
waitTexture = Screen('MakeTexture', wptr, imread(imgWait)); 
restTexture = Screen('MakeTexture', wptr, imread(imgRest)); 
endTexture = Screen('MakeTexture', wptr, imread(imgEnd)); 

%% Show start and wait instruction
Screen('DrawTexture', wptr, startTexture);
Screen('Flip', wptr);
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(insKey), break;
    end
end

Screen('DrawTexture', wptr, waitTexture);
Screen('Flip', wptr);
while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    if keyIsDown && keyCode(insKey), break;
    end
end
%% Run experiment: show stimui and wait response for each trial
% load design matrix
load(designMatrixMat);
paradigmClass = BIN.paradigmClass;
sessionLevel = 4*(subject.subID-1)+subject.sessionID;
%allocate 1000 trails into 10 run
paradigmClassReshape = reshape(paradigmClass(:,sessionLevel,:), 100, 10, 3);
stimOnsetAll = paradigmClassReshape(:,:, 1);
stimAll = paradigmClassReshape(:,:, 2);
resultPerSession = cell(10, 1);
% loop to run the experiment
for runIndex = 1:10
    % prepare stim name and onset
    stimRun = stimAll(:, runIndex);
    stimOnset = cell2mat(stimOnsetAll(:, runIndex)); 
    stimOnset = stimOnset - stimOnset(1,1); %substract the first time
    % make stimuli texture
    stimTexture = zeros(1,100);
    for trailIndex = 1:100
        picName = stimRun{trailIndex, 1};
        picClass = regexp(picName, '_', 'split');
        imgPath = sprintf('%s/%s/%s', stimulusFolder, picClass{1}, picName);
        imgResize = imresize(imread(imgPath), [imgPixelHor imgPixelVer]);
        stimTexture(trailIndex) = Screen('MakeTexture', wptr, imgResize);
    end
    % show null trials before stimulus
    Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor+5, [50 50 50], [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor, fixColor, [], 2);
    Screen('Flip', wptr);
    WaitSecs(nullBlank);    
  
    % start the experiment
    tStart = GetSecs;
    responseArr = cell(100, 3);
    trailIndex = 1;
    while 1
        % show fixation for all time
        Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor+5, [50 50 50], [], 2);
        Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor, fixColor, [], 2);
        Screen('Flip', wptr);
        % show stimulus at the right time
        stimOnsetSingle = stimOnset(trailIndex, 1);
        tCurrent = floor(GetSecs - tStart);
        if tCurrent == stimOnsetSingle          
            % ON Trail
            Screen('DrawTexture', wptr, stimTexture(trailIndex));
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
            responseArr{trailIndex, 1} = tCurrent;
            responseArr{trailIndex, 2} = response;
            responseArr{trailIndex, 3} = RT ;            
            trailIndex = trailIndex+1;% Move on trail
        end
        % using in debugging
        if strcmp(response, 'break')
            break;
        end
        if trailIndex > 100, break, end %break after 100 trails
    end
    
    % write response info result mat
    resultPerSession{runIndex} = cat(2, stimRun, responseArr);
    % using in debugging
    if strcmp(response, 'break')
        break;
    end
    % show null trials after stimulus
    Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor+5, [50 50 50], [], 2);
    Screen('DrawDots', wptr, [xCenter,yCenter], fixPixelHor, fixColor, [], 2);
    Screen('Flip', wptr);
    WaitSecs(nullBlank);    
    % Show rest instruction
    Screen('DrawTexture', wptr, restTexture);
    Screen('Flip', wptr);
    while KbCheck(); end
    while true
        [keyIsDown,~,keyCode] = KbCheck();
        if keyIsDown && keyCode(insKey), break;
        end
    end
end

%% Save data
outPath = sprintf('%s/sub%02d', outFolderName, subject.subID);
if ~exist(outPath,'dir')
    mkdir(outPath);
end
fprintf('Data were saved to: %s\n',outPath);
outName = sprintf('%s/session%02d.mat', outPath, subject.sessionID);
save(outName,'resultPerSession');

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
