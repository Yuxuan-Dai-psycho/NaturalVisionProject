function trial = RunBasedImageNetMEG(subID,runID)
% trial = RunBasedImageNetMEG(subID,runID)
% Wrap the ImageNetMEG with only subID and runID as input
trial = ImageNetMEG(subID,ceil(runID/5),mod(runID,5));

