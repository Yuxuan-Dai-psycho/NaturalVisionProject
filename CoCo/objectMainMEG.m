%%======！！！！请务必不要搞错subID 和 sessID ！！！=====
clear; 
close all
workDir = pwd;

%% Set subject and session info
% You should manually input subject ID and session ID
subName = 'Test';subID = 10086; sessID = 1; 

%% Run ImageNet MEG 
% For 10 core subjects, there two rounds of ImageNet MEG: 
% session 1 and 2 will be conducted in 1st round, and 
% session 3 and 4 will be conducted in the 2nd round
% For other 20 subjects, only one round of MEG exp: 
% session 1 and 2 will conducted.

% You should mannual change sessID and runID for each run
close all;sca;
objectImageNetMEG(subID,sessID,1);


%% Run ImageNet memroy ep 
% You should mannual change runID for each run
close all;sca;
objectImageNetMemory(subID,sessID);

%% Run CoCo MEG 
% For 10 core subjects, there is one session of COCO MEG. 
% For other 20 subjects, no COCO MEG

% You should mannual change runID for each run
close all;sca;
objectCoCoMEG(subID,sessID,1);


%% Run Resting MEG 
% You should mannual change runID for each run
close all;sca;
objectRestingMEG(subID,sessID);


%% Run CoCo memroy 
% You should mannual change runID for each run
close all;sca;
objectCoCoMemory(subID,sessID);
