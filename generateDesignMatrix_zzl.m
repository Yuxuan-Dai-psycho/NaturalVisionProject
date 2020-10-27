% This file is to generate design matrix
% Organize both stimulus image and stimulus order information into
% BrainImageNet(BIN) structure

%% Directory setting
stimDir =  '/nfs/e1/BrainImageNet/stim';
imgDir = fullfile(stimDir,'images');
designDir = fullfile(stimDir,'designMatrix');


%% Load super class  
% read super class info
fid = fopen(fullfile(designDir,'superClassMapping.csv'));
C = textscan(fid, '%s %d %s %d','Headerlines',1, 'Delimiter',',');
fclose(fid);
classID = C{1};
superClassID = C{2}; 
superClassName = C{3}; 

nClass = 1000; 
nSuperClass = 30;
nSession = 80;

%% Organize stimulus according to the super class info 
stimulus = cell(nClass,nSession);
for i = 1:length(classID) % class loop
    imageName = dir(fullfile(imgDir,classID{i})); 
    imageName = extractfield(imageName(3:end), 'name');
    stimulus(i,:) = imageName(randperm(length(imageName)));
    imageName = [];
end


%% Load optseq of super class 
optSeqSuperClass = NaN(nClass,nSession,3);% [onset, class, dur]
for s = 1:nSession % session loop
    % Read par from optseq
    optSeqSuperClassFile = fullfile(designDir,'ExpsessionorderTR1',...
        sprintf('BIN-static-session-%03d.csv',s));
    fid = fopen(file);
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
for s = 1:nSession % session loop
    for c = 1:nSuperClass % class loop
        superClassTrial = (optSeqSuperClass(:,s,2) == c-1);
        classTrial = find(superClassID == c-1);       
        optSeqClass(superClassTrial,s,2) = classTrial(randperm(length(classTrial))) ;
    end
end


%% Align onset of trials to the first trial of each run
nRun = 10;
for s = 1:nSession
    onset = reshape(optSeqClass(:,s,1),[],nRun);
    for r = 1:nRun 
        onset(:,r) = onset(:,r) - onset(1,r);
    end
    optSeqClass(:,s,1) = onset(:);
end


%% Pack and save BIN strcture
BIN.desp = 'BrainImageNet session-level paradigm';
BIN.classID = classID;
BIN.superClassName = superClassName;
BIN.superClassID = superClassID;
BIN.stimulus = stimulus;
BIN.paradigmSuperClass = optSeqSuperClass;
BIN.paradigmClass = optSeqClass;

% save BIN
% save('BIN.mat', 'BIN');
