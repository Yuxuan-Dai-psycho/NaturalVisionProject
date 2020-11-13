% The script generates design matrix for BrainImageNet test set
% Organize both stimulus image and stimulus order information into
% Test structure 
clc;clear;
%% basic setting
stimDir = 'H:\NaturalImageData\stimTest';
imgDir = fullfile(stimDir,'images');
nSub = 20; % 
nRun = 10; 




%% List stimulus 
imageName = dir(fullfile(imgDir)); 
imageName = extractfield(imageName(3:end), 'name');

for sub = 1:20
    stimulus(sub,:,:) = reshape(imageName(randperm(length(imageName))), 10, 12);
end

%% Load mSequence 
mSeqCondition = load(fullfile(stimDir,'testSeq.mat'));
stimName = stimulus(:,:,mSeqCondition(:,2));
stimOnset = permute(repmat(mSeqCondition(:,1),1,20,10),[2,3,1]);
mSeqStim(:,:,:,1) = num2cell(stimOnset);
mSeqStim(:,:,:,2) = stimName;

%% Pack and save BCC strcture
Test.desp = 'BrainCoCo run-level paradigm';
Test.stimulus = stimulus;
Test.mSeqCondition = mSeqCondition;
Test.mSeqStim = mSeqStim;

% save Test
save(fullfile(stimDir,'BIN.mat'), 'Test');
