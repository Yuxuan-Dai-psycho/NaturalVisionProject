# BrainImageNet
Brain imaging data for ImageNet

## Instructions
-----------------------------------
1. Download the dataset in the server: /nfs/e1/BrainImageNet/stim; Two folder：{images} & {DesignMatrix}
    Modify the workDir in binMRI.m(line 32) in your own computer.
    Remember stimDir and designDir are under workDir or you can change them by your own habbit.
2. Run binMRI.m to start the whole experiment. You need to enter the function with parameters(subID, sessID, runID).
3. When start instruction is onset, you need to press likeKey('F') to start. After that, the ready instruciton is onset and you need to press startKey('S') to start the experiment.