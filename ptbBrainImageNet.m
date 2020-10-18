function subject = ptbBrainImageNet()
% Present ImageNet images for each category and get fMRI signal
%
% Inputs (necessary):
%   

% Version 1.4 2020/10/17
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
stimOrderFolder = 'stim/designMatrix';
imgPixel = 800;
run_per_session = 10;
stimulus_onset = 2;
blank_Interval = 2;
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

%% Open a .txt file for saving the data
outPath = sprintf('%s/sub%d', outFolderName, subject.subID);
if ~exist(outPath,'dir')
    mkdir(outPath);
end
txtFileName_Result = sprintf('%s/expImage_sub%d_session%d.txt', outPath, subject.subID, subject.sessionID);
fid = fopen(txtFileName_Result, 'w+');
fclose(fid);

%% Run experiment: show stimui and wait response for each trial
for runIndex = 1:run_per_session
    % load stimulus
    stimPath = sprintf('%s/sub%d/session%d/stim_run%d.mat', stimOrderFolder, subject.subID, subject.sessionID, runIndex);
    stimStruct = load(stimPath);
    stimAll = stimStruct.stimAll;
    % make stimuli texture
    stimTexture = zeros(1,size(stimAll, 1));
    for trailIndex = 1:size(stimAll, 1) 
        picName = stimAll{trailIndex};
        if strcmp(picName, 'NA')
            imgPath = sprintf('%s/%s', picsFolderName, 'NA.jpg');
        else
            imgPath = sprintf('%s/%s', stimulusFolder, picName);
        end
        imgResize = imresize(imread(imgPath), [imgPixel imgPixel]);
        stimTexture(trailIndex) = Screen('MakeTexture', wptr, imgResize);
    end
    
     % loop to run the experiment
     for trailIndex = 1:size(stimAll, 1) 
   
        % Show the corresponding stimuli
        Screen('DrawTexture', wptr, stimTexture(trailIndex));
        Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
        tStart = Screen('Flip',wptr);   
        
        % Wait response
        while KbCheck(), end % empty the key buffer
        while GetSecs - tStart < stimulus_onset
            [~, ~, keyCode] = KbCheck();
            
            if keyCode(escKey), response = 'break'; break;
            elseif keyCode(likeKey),   response = 1;
            elseif keyCode(disLikeKey), response = 0;
            else, response = -1;
            end
                
        end   
        
        % Show the fixation  
        Screen('DrawDots', wptr, [xCenter,yCenter], fixSize, fixColor, [], 2);
        Screen('Flip', wptr);
        WaitSecs(blank_Interval);    
        
        % using in debugging
        if strcmp(response, 'break')
            break;
        end
        
        % write response info into output txt
        tmpArr = [subject.subID subject.sessionID trailIndex response];
        tmpLine = sprintf('%d,%d,%d,%d,%d,%s', tmpArr, picName);

        fid = fopen(txtFileName_Result, 'a+');
        fprintf(fid, '%s\r\n', tmpLine);
        fclose(fid);        
    end

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
% outFile = fullfile('data',sprintf('%s_%s_stroop.mat',subject.name,task.date));
% fprintf('Test data were saved to: %s\n',outFile);
% save(outFile,'subject','task');

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

end
