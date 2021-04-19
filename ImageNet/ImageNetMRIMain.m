%% Please run exp under the home dir of NaturalVisionProject
clear; close all
workDir = pwd;
addpath(genpath(workDir));

%% Set subject and session info: 请务必不要搞错subID 和 sessID  
% You should manually input subject ID and session ID
% % sub11\13\14\18\19 -sess01 has been scanned 
% correspond to 11\19\23\39\43th subject in the new design
subName = 'Test';subID = 10086; sessID = 1; 

% You should manually input subject ID and run ID for MEG
subName = 'Test';subID = 10086; runID = 1; % run ID should be a integer within [1:20] for Sub ID 1-10, 
                                           % [1:10] for Sub ID 11-30!
                                     
%% Run ImageNet fMRI  
% You should mannual change runID for each run
close all;sca;
ImageNetMRI(subID,sessID,runID);

%% Run Resting fMRI  
% You should mannual change runID for each run
close all;sca;
ImageNetRestingMRI(subID,sessID);

%% Run ImageNet memroy 
% You should mannual change runID for each run
close all;sca;
ImageNetMemory(subID,sessID);


