clear; clc;
%% Generate design matrix for MEG training
workDir = './stim';
imgDir = fullfile(workDir, 'images');
designDir = fullfile(workDir,'MEGdesignMatrix');

%% Load file names of images and shuffle
% TODO
classNames = {dir(imgDir).name};
classNames = classNames(3:end);
imageFiles = cell(1, 1000);
for i = 1: 1000
    files = {dir(fullfile(imgDir, classNames{i})).name};
    imageFiles{i} = files(3:end);
    imageFiles{i} = imageFiles{i}(randperm(80));
end

nSession = 40;
nRun = 8;
nTrial = 300;

%% Define the design matrix
% 'stim' for file names of images
% 'class' for class names of images
% 'type' for trial type: 1 for normal, 0 for badge detection trial
% 'accumTrialTime' for absolute accumulated time
% 'fixationTime' for fixation time of each trial
MEGDesignMatrix = struct();
MEGDesignMatrix.stim = cell(nSession, nRun, nTrial);
MEGDesignMatrix.class = cell(nSession, nRun, nTrial);
MEGDesignMatrix.type = zeros(nSession, nRun, nTrial);
MEGDesignMatrix.accumTime = zeros(nSession, nRun, nTrial);
MEGDesignMatrix.fixationTime = rand(nSession, nRun, nTrial) * 0.1 + 0.4;

%% Determine type of each trial
for i = 1: nSession
    for j = 1: nRun
        while true
            types = [];
            while length(types) < nTrial
                if nTrial - length(types) < 4
                    types = [types, ones(1, nTrial - length(types))];
                else
                    interval = randi(3) + 3; % for {4, 5, 6}
                    types = [types, ones(1, interval), 0];
                end
            end
            if (length(types) == nTrial) && sum(types) == 250
                MEGDesignMatrix.type(i, j, :) = types;
                break;
            end
        end
    end
end

%% Determine stimulus for each trial
for i = 1: nSession
    sessImg = cell(1, 2000);
    sessOrder = randperm(2000);
    for idx = 1: 2000 % 2000 images per session
        sessImg{idx} = imageFiles{ceil(idx / 2)}{2 * i + mod(idx+1, 2) - 1};
    end
    sessImg = sessImg(sessOrder);

    for j = 1: nRun
        for k = 1: nTrial
            % 0.5 for duration of each stimulus
            MEGDesignMatrix.accumTime(i, j, k) = sum(0.5 + MEGDesignMatrix.fixationTime(i, j, 1:k));
            if MEGDesignMatrix.type(i, j, k) % Normal image
                nImageTrial = sum(MEGDesignMatrix.type(i, j, 1: k));
                img = sessImg{(j-1)*nRun + nImageTrial};
                cls = img(1: strfind(img, '_')-1);
                MEGDesignMatrix.stim{i, j, k} = img;
                MEGDesignMatrix.class{i, j, k} = cls;
            else % Badge detection
                MEGDesignMatrix.stim{i, j, k} = 'badge.jpg';
                MEGDesignMatrix.class{i, j, k} = '../../instruction/';
            end
        end
    end
end

%% Export design matrix
if ~isdir(designDir), mkdir(designDir); end
save(fullfile(designDir, 'MEGDesignMatrix.mat'), 'MEGDesignMatrix');