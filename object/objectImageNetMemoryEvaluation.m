function objectImageNetMemoryEvaluation(subID,sessID)
% function objectImageNetMemoryEvaluation(subID,sessID)
% Evaluate memory performance from after fmri experiment
workDir = pwd;
% Create session dir
sessDir = fullfile(workDir, 'data','fmri','train',...
    sprintf('sub%02d/sess%02d',subID,sessID));
figure('Name',sprintf('sub%02d-sess%02d',subID,sessID),...
    'Units','normalized','Position',[0 0 1 1])
for runID = 1:4
    fileName =fullfile(sessDir,...
        sprintf('sub%02d_sess%02d_run%02d_beh.mat',subID,sessID,runID));
    if ~exist(fileName,'file')
        warning([fileName,' ', 'Not Exsit']);
        continue
    end
    
    % Load data
    load(fileName,'trial'); % [onset, categoryID, cond, key, rt]
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
    idx = any(response,2);% only keep trial with response
    [cVal,cMat,~,cPer] = binConfusion(target(idx,:)',response(idx,:)');
    subplot(1,4,runID),
    imagesc(cMat);
    title(sprintf('RespProp = %.2f, Accuracy = %.2f',sum(idx)/length(target) ,1-cVal));
    axis square; colorbar
    condName = {'Seen', 'Unseen'};
    set(gca,'Xtick',1:length(cMat), 'XTickLabel',condName,...
        'Ytick',1:length(cMat),'YTickLabel',condName);
    text(0.65,1,sprintf('%.2f',cPer(1,3)),'FontSize',25,'Color','r');% hit
    text(0.65,2,sprintf('%.2f',cPer(1,1)),'FontSize',25,'Color','r');% miss
    text(1.65,1,sprintf('%.2f',cPer(1,2)),'FontSize',25,'Color','r');% false alarm
    text(1.65,2,sprintf('%.2f',cPer(1,4)),'FontSize',25,'Color','r');% corect reject
end
% Save figure
figureFile = fullfile(sessDir,sprintf('sub%02d_sess%02d_beh.jpg',subID,sessID));
print(figureFile,'-djpeg');