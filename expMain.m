%%======！！！！IMPORTANT 注意检查被试编号与session号！！！！=====
clear; 
close all

%% Set subject and session info
% You should manually input subject ID and session ID
subName = 'shenhuadong';subID = 4; sessID = 1; 

%% Run BIN train exp 
% You should mannual change runID for each run
binMRItrain(subID,sessID,1);clc;sca;

%% Run BIN test exp 
% You should mannual change runID for each run
binMRItest(subID,sessID,1);clc;sca;

