%%======！！！！请务必不要搞错subID 和 sessID ！！！=====
clear; 
close all
workDir = pwd;

%% Set subject and session info
% You should manually input subject ID and session ID
subName = 'shenhuadong';subID = 4; sessID = 1; 

%% Run BIN ImageNet fMRI exp: 
% You should mannual change runID for each run
close all;sca;
objectImageNetMRI(subID,sessID,1);


%% Run BIN ImageNet memroy exp 
% You should mannual change runID for each run
close all;sca;
objectImageNetMemory(subID,sessID);


%% Run BIN CoCo fMRI exp 
% You should mannual change runID for each run
close all;sca;
objectCocoMR(subID,sessID,1);


%% Run BIN CoCo memroy exp 
% You should mannual change runID for each run
close all;sca;
objectCoCoMemory(subID,sessID);
