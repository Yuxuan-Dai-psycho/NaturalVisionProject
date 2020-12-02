%%======！！！！请务必不要搞错subID 和 sessID ！！！=====
clear; 
close all
worDir = '/codeDir/in/your/computer';
cd(workDir)

%% Set subject and session info
% You should manually input subject ID and session ID
subName = 'shenhuadong';subID = 4; sessID = 1; 

%% Run BIN train exp 
% You should mannual change runID for each run
binMRItrain(subID,sessID,1);clc;sca;

%% Run BIN test exp 
% You should mannual change runID for each run
binMRItest(subID,sessID,1);clc;sca;

