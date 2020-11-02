# BrainImageNet
Brain imaging data for ImageNet

## Instructions
-----------------------------------
1. Download the dataset in the server: /nfs/e1/BrainImageNet/stim. Modify the "workDir" in binMRI.m(line 32) according to your download path.
2. Run binMRI.m to start the whole experiment. You need to enter the function with parameters(subID, sessID, runID).
3. When start instruction is onset, you need to press likeKey('F') to skip. After that, the ready instruciton is onset and you need to press startKey('S') to start the experiment.