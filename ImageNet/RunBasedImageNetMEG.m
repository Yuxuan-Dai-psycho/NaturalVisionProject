function trial = RunBasedImageNetMEG(subID,runID)
% trial = RunBasedImageNetMEG(subID,runID)
% Wrap the ImageNetMEG with only subID and runID as input
if mod(runID,5) == 0
    trial = ImageNetMEG(subID,ceil(runID/5),5);
else
    trial = ImageNetMEG(subID,ceil(runID/5),mod(runID,5));
end

