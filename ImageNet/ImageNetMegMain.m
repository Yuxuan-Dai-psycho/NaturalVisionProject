%% Please run exp under the home dir of NaturalVisionProject
clear; close all
workDir = pwd;
addpath(genpath(workDir));

%% Set subject and session info: ����ز�Ҫ���subID �� sessID  
subID = 1; runID = 1;
                                           
%% Run ImageNet MEG 
% You should mannual change sessID and runID for each run
% For 10 core subjects, there are 20 runs: 
% For other 20 subjects, there are 5 runs: 
close all;sca;
RunBasedImageNetMEG(subID,runID)
                                   

