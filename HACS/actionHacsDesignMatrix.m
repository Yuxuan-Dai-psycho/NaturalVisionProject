% The script generates design matrix for action HACS dataset
% Organize both stimulus video and stimulus order information into
% action structure

clc;clear;
%% Directory setting
stimDir =  '/nfs/m1/BrainImageNet/action';
videoDir = fullfile(stimDir,'video');
designDir = fullfile(stimDir,'designMatrix');

%% Load super class  
% read super class info
fid = fopen(fullfile(designDir,'actionTaxonomy.csv'));
C = textscan(fid, '%d %s %d %s','Headerlines',1, 'Delimiter',',');
fclose(fid);
classID = C{1}; % action class id, 200x1, int array
className = C{2}; % action class name, 200x1, cell array
superClassID = C{3}; % 14 superClass ID, 200x1, int array, 
superClassName = C{4}; % super class name, 200x1 cell array

nClass = 180;
nSuperClass = 13;
nSession = 120;
nRepeat = 4; % repeat times of all class in one session
nSessionTmp = nSession*nRepeat; 
nStimPerSession = nClass*nRepeat;

%% Organize stimulus according to the super class info 
stimulus = cell(nClass,nSessionTmp); % 180x480 cell array
for i = 1:length(className) % class loop
    videoName = dir(fullfile(videoDir,className{i})); 
    videoName = extractfield(videoName(3:end), 'name');
    stimulus(i,:) = videoName(randperm(length(videoName)));
    videoName = [];
end

%% Load optseq of super class 
optSeqSuperClass = NaN(nClass,nSessionTmp,3);% [onset, class, dur]
for s = 1:nSessionTmp % session loop
    % Read par from optseq
    optSeqSuperClassFile = fullfile(designDir,'sessPar',...
        sprintf('action-session-%03d.par',s));
    fid = fopen(optSeqSuperClassFile);
    optSeq = textscan(fid, '%d %d %d %d %s');
    fclose(fid);

    % Remove null event and assemble optSeqSuperClass
    optSeq = cell2mat(optSeq(1:3));
    optSeq = optSeq(optSeq(:,2) ~= 0,:);
    optSeqSuperClass(:,s,:) = optSeq;
    optSeq = [];
end

%% Translate superClass optSeq to class optSeq
optSeqClass = optSeqSuperClass;
for s = 1:nSessionTmp % session loop
    for c = 1:nSuperClass % class loop
        superClassTrial = (optSeqSuperClass(:,s,2) == c);
        classTrial = find(superClassID == c);       
        optSeqClass(superClassTrial,s,2) = classTrial(randperm(length(classTrial))) ;
    end
end
% check output: eg=squeeze(optSeqClass(:,1,:));

%% Reshape stimulus and design sequence as required
% reshape stimulus to 720x120 array
% reshape optSeqSuperClass&optSeqClass to 720x120x3 array
stimulus = reshape(stimulus,nStimPerSession,nSession); 
optSeqClass = reshape(optSeqClass,nStimPerSession,nSession,3); 
optSeqSuperClass = reshape(optSeqSuperClass,...
        nStimPerSession,nSession,3); 

%% Replace optSeq timing with Kay design
nRun = 12; runDur = 284; trialDur = 4; % in seconds
kaySeq = 0:trialDur:runDur; % Trials are uniformly-spaced in time(i.e., 4s)
kaySeq(6:6:length(kaySeq)) = []; % Remove null trials(evey sixth trials)
onset = repmat(kaySeq, [1,nRun]);
for s = 1:nSession
    optSeqClass(:,s,1) = onset;
    optSeqSuperClass(:,s,1) = onset;  
end
 
%% Pack and save action strcture
action.desp = 'actionHacs session-level paradigm';
action.className = className; % HACS class name, 180x1, cell array
action.superClassName = superClassName;% HACS superclass id, 180x1, cell array
action.superClassID = superClassID;% HACS superclass class id, 180x1, int array
action.stimulus = stimulus; % 720 x 120, cell array
action.paradigmSuperClass = optSeqSuperClass; % 720(nClass*4) x 120(session) x 3 array
action.paradigmClass = optSeqClass; % 720(nClass*4) x 120(session) x 3  array
action.date = datetime; 

% Save action to design dir
save(fullfile(designDir,'action.mat'), 'action');
