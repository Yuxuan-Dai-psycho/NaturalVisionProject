%% Please run exp under the home dir of NaturalVisionProject
clear; close all
workDir = pwd;
addpath(genpath(workDir));

%% Set subject and session info: 请务必不要搞错subID 和 sessID  
% You should manually input subject ID and session ID for fMRI
subName = 'Test';subID = 10086; sessID = 1; 

% You should manually input subject ID and run ID for MEG
subName = 'Test';subID = 10086; runID = 1; % run ID should be a integer within [1:10]!
sessID = 1; % No manual changes are required

%% Run CoCo fMRI  
% You should mannual change runID for each run
close all;sca;
CoCoMRI(subID,sessID,1);

%% Run Resting fMRI  
% You should mannual change runID for each run
close all;sca;
RestingMRI(subID,sessID);

%% Run CoCo memroy  
% You should mannual change runID for each run
close all;sca;
CoCoMemory(subID,sessID);
