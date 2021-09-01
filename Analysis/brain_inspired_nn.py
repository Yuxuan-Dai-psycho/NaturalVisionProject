import torch
import pickle as pkl
import torch.nn.functional as F

from torch import nn
from os.path import join as pjoin
from model_utils import CustomizedLinear

# define network
n_class = 10 # the classification class num
class VisualNet(nn.Module):
    def __init__(self):
        super(VisualNet, self).__init__()
        # load mask info
        # define path
        main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
        out_path = pjoin(main_path, 'imagenet_decoding')
        # load brain structure info
        with open(pjoin(out_path, 'dnn_struct_info.pkl'), 'rb') as f:
            dnn_struct_info = pkl.load(f)
        # load mask
        voxel2roi_mask = torch.tensor(dnn_struct_info['voxel2roi_mask'])
        roi2network_mask = torch.tensor(dnn_struct_info['roi2network_mask'])
        del dnn_struct_info
        
        # define layer
        self.voxel2roi = CustomizedLinear(voxel2roi_mask)
        self.roi2network = CustomizedLinear(roi2network_mask)
        self.output = nn.Linear(roi2network_mask.size()[1], n_class)
        
    def forward(self, x):
        x = self.voxel2roi(x)
        x = self.roi2network(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)  
        return x
        