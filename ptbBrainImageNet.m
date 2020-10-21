% Present ImageNet images for each category and get fMRI signal
%

% Version 1.4 2020/10/21
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

%% Prepare foldernames and params
picsFolderName = 'instruction';
outFolderName = 'out';
stimulusFolder = 'stim/images';
designMatrixMat = 'stim/BIN.mat';
imgPixel = 800;
blank_Interval = 2;
run_per_session = 10;
fixSize = 10;
fixColor = [255 255 255];

%% Response keys setting 
PsychDefaultSetup(2);% Setup PTB to 'featureLevel' of 2
KbName('UnifyKeyNames'); % For cross-platform compatibility of keynaming
insKey = KbName('s');
likeKey = KbName('f');
disLikeKey = KbName('j'); 
escKey   = KbName('escape'); % stop and exit

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
    if keyIsDown && keyCode(insKey), break;
    end
end

%% Run experiment: show stimui and wait response for each trial
% load design matrix
load(designMatrixMat);
paradigmClass = BIN.paradigmClass;
sessionLevel = 4*(subject.subID-1)+subject.sessionID;
stimPerSession = paradigmClass{sessionLevel};
resultPerSession = cell(run_per_session, 1);

% loop to run the experiment
for runIndex = 1:run_per_session
    % generate run stim matrix
    tmpDelete = stimPerSession;
    tmpDelete(find(strcmp(tmpDelete, 'NULL')), :) = [];
    imageSeperate = tmpDelete{100, 1};
    stimRow = find(strcmp(stimPerSession, imageSeperate));
    if runIndex ~= run_per_session
        stimRun = stimPerSession(1:stimRow,:);
    else
        stimRun = stimPerSession;
    end
    stimPerSession = stimPerSession((stimRow+1):size(stimPerSession,1),:);
    % cantante null trail   
    nullTrails = {'NULL', 16};
    stimRun = cat(1, nullTrails, stimRun, nullTrails);
    % make stimuli texture
    stimTexture = zeros(1,size(stimRun, 1));
    for trailIndex = 1:size(stimRun, 1) 
        picName = stimRun{trailIndex, 1};
        if strcmp(picName, 'NULL')
            imgPath = sprintf('%s/%s', picsFolderName, 'NULL.jpg');
        else
            imgPath = sprintf('%s/%s', stimulusFolder, picName);
        end
        imgResize = imresize(imread(imgPath), [imgPixel imgPixel]);
        stimTexture(trailIndex) = Screen('MakeTexture', wptr, imgResize);
    end
    
     % loop to run the experiment
     responseArr = cell(size(stimRun, 1), 2);
     for trailIndex = 1:size(stimRun, 1) 
        picName = stimRun{trailIndex, 1};
        stimDur = stimRun{trailIndex, 2};
        % Show the corresponding stimuli
        Screen('DrawTexture', wptr, stimTexture(trailIndex));
        Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
        tStart = Screen('Flip',wptr);   
        
        % Wait response
        pressTime = 0;
        response = -1;
        RT = stimDur;
        while KbCheck(), end % empty the key buffer
        while GetSecs - tStart < stimDur-blank_Interval
            [keyIsDown, tEnd, keyCode] = KbCheck();
            if keyIsDown
                if pressTime < 1
                    if keyCode(escKey), response = 'break'; break;
                    elseif keyCode(likeKey),   response = 1; RT = tEnd-tStart;
                    elseif keyCode(disLikeKey), response = 0; RT = tEnd-tStart;
                    end
                end
                pressTime = pressTime + 1;
            end
        end
        % record response
        responseArr{trailIndex, 1} = response;
        responseArr{trailIndex, 2} = RT ;
            
        % using in debugging
        if strcmp(response, 'break')
            break;
        end
        
        % Show blank interval
        if ~strcmp(picName, 'NULL')
            Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
            Screen('Flip', wptr);
            WaitSecs(blank_Interval);           
        end
    end
    % write response info result mat
    resultPerRun = cat(2, stimRun, responseArr);
    resultPerSession{runIndex} = resultPerRun;
       
    % Show rest instruction
    Screen('DrawTexture', wptr, restTexture);
    Screen('Flip', wptr);
    while KbCheck(); end
    while true
        [keyIsDown,~,keyCode] = KbCheck();
        if keyIsDown && keyCode(insKey), break;
        end
    end
    % using in debugging
    if strcmp(response, 'break')
        break;
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
