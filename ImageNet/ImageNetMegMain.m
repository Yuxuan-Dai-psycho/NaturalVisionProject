%% Please run exp under the home dir of NaturalVisionProject
clear; close all
workDir = pwd;
addpath(genpath(workDir));

%% Set subject and session info: 请务必不要搞错subID 和 sessID  
subID = 1; runID = 1;
                                           
%% Run ImageNet MEG 
% You should mannual change sessID and runID for each run
% For 10 core subjects, there are 20 runs: 
% For other 20 subjects, there are 7 runs: 
close all;sca;
RunBasedImageNetMEG(subID,runID)
                                   

