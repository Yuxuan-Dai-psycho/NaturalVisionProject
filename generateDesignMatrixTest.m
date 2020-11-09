% This file is to generate design matrix for Test set
% Organize both stimulus image and stimulus order information into
% BrainCoCo(BCC) structure

clc;clear;
%% Directory setting
stimDir = '/nfs/e1/BrainImageNet/stimTest';
imgDir = fullfile(stimDir,'SelectedImagesCOCO');
designMat = fullfile(stimDir,'testSeq.mat');

%% Prepare stimulus 
imageName = dir(fullfile(imgDir)); 
imageName = extractfield(imageName(3:end), 'name');
for sub = 1:20
    stimulus(sub,:,:) = reshape(imageName(randperm(length(imageName))), 10, 12);
end

%% Generate mSequence stim
load(designMat);
stimName = stimulus(:,:,mSeqCondition(:,2));
stimOnset = permute(repmat(mSeqCondition(:,1),1,20,10),[2,3,1]);
mSeqStim(:,:,:,1) = num2cell(stimOnset);
mSeqStim(:,:,:,2) = stimName;

%% Pack and save BCC strcture
BCC.desp = 'BrainCoCo run-level paradigm';
BCC.stimulus = stimulus;
BCC.mSeqCondition = mSeqCondition;
BCC.mSeqStim = mSeqStim;

% save BCC
save(fullfile(stimDir,'BCC.mat'), 'BCC');
