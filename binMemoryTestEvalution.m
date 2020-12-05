function binMemoryTestEvalution(subID,sessID)
% function binMemoryTestEvalution(subID,sessID)
% Evaluate memory performance from after fmri experiment

% Create session dir
sessDir = fullfile('data','fmri','train',...
    sprintf('sub%02d/sess%02d',subID,sessID));

for runID = 1:4
    fileName =fullfile(sessDir,...
        sprintf('sub%02d_sess%02d_run%02d_beh.mat',subID,sessID,runID));
    if ~exist(fileName,'file')
        warning([fileName,' ', 'Not Exsit']);
        continue
    end
    
    % Load data
    load(fullfile(sessDir,fileName),'trial'); % [onset, categoryID, cond, key, rt]
    nTrial = length(trial);
    
    % Make target matrix nTrial x nCond
    target = zeros(nTrial,2);
    target(:,1) = trial(:,3) == 1; % appear in fMRI
    target(:,2) = trial(:,3) == 0; % not appear in fMRI
    
    % Make response matrix nTrial x nCond
    response = zeros(nTrial,2);
    response(:,1) = trial(:,4) == 1; % seen
    response(:,2) = trial(:,4) == -1; % unseen
    
    % Summarize the response with figure
    binResponseEvaluation(target,response,{'Seen', 'Unseen'});
    
    % Save figure
    figureFile = fullfile(sessDir,...
        sprintf('sub%02d_sess%02d_run%02d_beh.jpg',subID,sessID,runID));
    print(figureFile,'-djpeg');
end