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
className = C{3}; 

%% Organize stimulus according to the super class info 
stimulus = cell(1000,80);
for i = 1:length(classID) % class loop
    imageName = dir(fullfile(imgDir,classID{i})); 
    imageName = extractfield(imageName(3:end), 'name');
    stimulus(i,:) = imageName(randperm(length(imageName)));
    imageName = [];
end

%% Load optseq of super class 
optSeqSuperClass = NaN(1000,80,3);
for s = 1:80 % session loop
    % read par from optseq
    optSeqSuperClassFile = fullfile(designDir,'ExpsessionorderTR1',...
        sprintf('BIN-static-session-%03d.csv',s));
    fid = fopen(optSeqSuperClassFile);
    optSeq = textscan(fid, '%d %d %d %d %s');
    fclose(fid);

    % remove null event and assemble optSeqSuperClass
    optSeq = cell2mat(optSeq(1:3));
    optSeq = optSeq(optSeq(:,2) ~= 0,:);
    d(s) = length(optSeq);
    optSeqSuperClass(:,s,:) = optSeq;
    optSeq = [];
end

%% Translate superClass optSeq to class optSeq
optSeqClass = NaN(size(optSeqSuperClass));
optSeqClass(:,:,[1,3]) = optSeqSuperClass(:,:,[1,3]);
optSeqClass = num2cell(optSeqClass);

for s = 1:80 % session loop
    for c = 1:30 % class loop
        superClassTrial = optSeqSuperClass(:,s,2) == c;
        classTrail = find(superClassID == c); 
        classTmpID = classID(classTrail);
        optSeqClass(superClassTrial,s,2) = classTmpID(randperm(length(classTmpID))) ;
    end
    stimulusPerSession = stimulus(:,s);
end

%% Reshape optSeqClass to the new session design
optSeqName = num2cell(optSeqClass);
for s = 1:nSession % session loop
    optSeqName(:,s,2) = stimulus(optSeqClass(:,s,2),s);
end
optSeqName = reshape(optSeqName, 800, 100, 3);


%% Pack and save BIN strcture
superClassNamePath = fullfile(stimDir,'designMatrix/superClassName.mat');
load(superClassNamePath);
BIN.desp = 'BrainImageNet session-level paradigm';
BIN.classID = classID;
BIN.superClassName = superClassName;
BIN.superClassID = superClassID;
BIN.stimulus = stimulus;
BIN.paradigmSuperClass = optSeqSuperClass;
BIN.paradigmClass = optSeqClass;

% save BIN
save('BIN.mat', 'BIN');
