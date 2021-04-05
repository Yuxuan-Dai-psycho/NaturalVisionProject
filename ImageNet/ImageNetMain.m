%% Please run exp under the home dir of NaturalVisionProject
clear; close all
workDir = pwd;
addpath(genpath(workDir));

%% Set subject and session info: 请务必不要搞错subID 和 sessID  
% You should manually input subject ID and session ID
% % sub11\13\14\18\19 -sess01 has been scanned 
% correspond to 11\19\23\39\43th subject in the new design
subName = 'Test';subID = 10086; sessID = 1; 

%% Run ImageNet fMRI  
% You should mannual change runID for each run
close all;sca;
ImageNetMRI(subID,sessID,1);

%% Run Resting fMRI  
% You should mannual change runID for each run
close all;sca;
ImageNetRestingMRI(subID,sessID);

%% Run ImageNet memroy 
% You should mannual change runID for each run
close all;sca;
ImageNetMemory(subID,sessID);

%% Run ImageNet MEG 
% You should mannual change sessID and runID for each run
% For 10 core subjects, there two rounds of ImageNet MEG: 
% session 1 and 2 will be conducted in 1st round, and 
% session 3 and 4 will be conducted in the 2nd round
% For other 20 subjects, only one round of MEG exp: 
% session 1 and 2 will conducted.
close all;sca;
ImageNetMEG(subID,sessID,1);

