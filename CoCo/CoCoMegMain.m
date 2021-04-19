%% Please run exp under the home dir of NaturalVisionProject
clear; close all
workDir = pwd;
addpath(genpath(workDir));

%% Set subject and session info: 请务必不要搞错subID 和 sessID  
% You should manually input subject ID and session ID for fMRI
subID = 10086; runID = 1; % run ID should be a integer within [1:10]!

%% Run CoCo MEG 
% % You should mannual change runID for each run
% For 10 core subjects, there is one session of COCO MEG. 
% For other 20 subjects, no COCO MEG
close all;sca;
CoCoMEG(subID,1,runID);

