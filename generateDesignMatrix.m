% This file is to generate design matrix
%% Generate stimulus matrix
clear;clc
% load data
stimulus = {};
superClassMappingPath = 'stimulus.csv';
stimAll = readtable(superClassMappingPath, 'Delimiter', ',');

for sub = 1:20
    for session = 1:4
        matrix = stimAll(stimAll{:,1}==sub & stimAll{:,2}==session, :);
        stimID = matrix{:,3};
        stimulus = [stimulus stimID];
    end
end

%save mat
save('stimulus.mat', 'stimulus')

%% Generate optseq superclass matrix
clear;clc
% define path
optSeqSuperClass = cell(80,1);
optSeqFolder = 'ExpsessionorderTR1';

for sessionLevel = 1:80
    optSeqSuperClassName = sprintf('%s/BIN-static-session-%03d.csv', optSeqFolder, sessionLevel);
    optSeqRaw = readtable(optSeqSuperClassName);
    optSeq = optSeqRaw{:,[2,3]};
    optSeqSuperClass{sessionLevel} = optSeq;
end

%save mat
save('optSeqSuperClass.mat', 'optSeqSuperClass')

%% Generate optseq class matrix
clear;clc
optSeqClass = cell(80,1);
load('optSeqSuperClass.mat');
load('stimulus.mat');

superClassMappingPath = 'superClassMapping.csv';
classMapping = readtable(superClassMappingPath, 'Delimiter', ',');
classMapping{:, 2} = classMapping{:, 2} + 1;

for sessionLevel = 1:size(optSeqSuperClass, 1)
    % load session level class and super class
    classMappingSingle = table2cell(classMapping);
    optSeqSuperClassPerSession = optSeqSuperClass{sessionLevel, 1};
    optSeqClassPerSession = num2cell(optSeqSuperClassPerSession);
    stimulusPerSession = stimulus(:, sessionLevel);
    stimulusPerSessionClassName = cell(size(stimulusPerSession));
    for trail  = 1:size(stimulusPerSession, 1)
       imageID =  stimulusPerSession{trail, 1};
       imageSplit = regexp(imageID, '/', 'split');
       stimulusPerSessionClassName{trail, 1} = imageSplit{1};
    end
    % replace super class with class id in class cell
    for trailLevel = 1:size(optSeqSuperClassPerSession, 1)
        superClass = optSeqSuperClassPerSession(trailLevel, 1);
        if superClass == 0
            optSeqClassPerSession{trailLevel, 1} = 'NULL';
        else
            % random pick one image
            classCorr = classMappingSingle(cell2mat(classMappingSingle(:, 2)) == superClass, 1);
            randomPick = randperm(size(classCorr,1), 1);
            classID = classCorr{randomPick,1};
            classDeleteRow = find(strcmp(classMappingSingle(:, 1), classID));
            classMappingSingle(classDeleteRow, :) = [];
            % Add imageID in the cell
            imageRow = find(strcmp(stimulusPerSessionClassName, classID));
            imageID = stimulusPerSession{imageRow,1};
            optSeqClassPerSession{trailLevel, 1} = imageID;           
        end
    end
    optSeqClass{sessionLevel} = optSeqClassPerSession;
end

% save mat
save('optSeqClass.mat', 'optSeqClass');

%% Generate BrainImageNet struct
clear;clc
% load mat
load('optSeqSuperClass.mat');
load('optSeqClass.mat');
load('stimulus.mat');
load('superClassName.mat');

superClassMappingPath = 'superClassMapping.csv';
classMapping = readtable(superClassMappingPath, 'Delimiter', ',');
classMapping{:, 2} = classMapping{:, 2} + 1;
classMappingCell = table2cell(sortrows(classMapping, 'Var4'));
className = classMappingCell(:, 3);
superClass = classMappingCell(:, 2);

% construct BIN struct
BIN = struct();
BIN.desp = 'BrainImageNet session-level paradigm';
BIN.className = className;
BIN.superClassName = superClassName;
BIN.superClass = superClass;
BIN.stimulus = stimulus;
BIN.paradigmSuperClass = optSeqSuperClass;
BIN.paradigmClass = optSeqClass;

% save BIN
save('BIN.mat', 'BIN');
