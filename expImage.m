% the whole experiment 
function expImage(sub, session)
%% Prepare foldernames and params
picsFolderName = 'Pics';
outFolderName = 'out';
stimulusFolder = 'SelectedImages';
stimOrderFolder = 'stim';
imgPixel = 800;
run_per_session = 10;

%% Screen setting
% Skip synIRF tests
Screen('Preference', 'SkipSyncTests', 2);
Screen('Preference','VisualDebugLevel',4);
Screen('Preference','SuppressAllWarnings',1);
bkgColor = [128 128 128];
[wptr, rect] = Screen('OpenWindow', 0, bkgColor);
% close cursor
HideCursor;

%% Show the start image
imgStart = sprintf('%s/%s', picsFolderName, 'Instruction_Start.jpg');
startTexture = Screen('MakeTexture', wptr, imread(imgStart)); 
Instruction(wptr, startTexture);

%% Open a .txt file for saving the data
outPath = sprintf('%s/sub%d', outFolderName, sub);
if ~exist(outPath,'dir')
    mkdir(outPath);
end
txtFileName_Result = sprintf('%s/expImage_sub%d_session%d.txt', outPath, sub, session);
fid = fopen(txtFileName_Result, 'w+');
fclose(fid);

%% Run experiment: show stimui and wait response for each trial
for runIndex = 1:run_per_session
    % load stimulus
    stimPath = sprintf('%s/sub%d/session%d/stim_run%d.mat', stimOrderFolder, sub, session, runIndex);
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
        response = singleTrial(wptr, rect, stimTexture(trailIndex));
        % using in debugging
        if strcmp(response, 'break')
            break;
        end
        
        % write response info into output txt
        tmpArr = [sub session runIndex trailIndex response];
        tmpLine = sprintf('%d,%d,%d,%d,%d,%s', tmpArr, picName);

        fid = fopen(txtFileName_Result, 'a+');
        fprintf(fid, '%s\r\n', tmpLine);
        fclose(fid);        
    end

    % show the rest image
    imgStart = sprintf('%s/%s', picsFolderName, 'Instruction_Rest.jpg');
    restTexture = Screen('MakeTexture', wptr, imread(imgStart)); 
    Instruction(wptr, restTexture);
    % using in debugging
    if strcmp(response, 'break')
        break;
    end
end
    
%% Show the end image 
imgEnd = sprintf('%s/%s', picsFolderName, 'Instruction_Bye.jpg');
endTexture = Screen('MakeTexture', wptr, imread(imgEnd)); 
Instruction(wptr, endTexture);

% show cursor and close all
ShowCursor;
Screen('CloseAll');

end
