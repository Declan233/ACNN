from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
from torch import nn
import numpy as np

class ACNN_DPAv4(nn.Module):
    """ACNN model for DPAv4"""
    def __init__(self, nbclass:int):
        super(ACNN_DPAv4, self).__init__()
        self.net_architecture = nn.Sequential(
            # Feature blocks
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=51, stride=1, padding=25),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=25, stride=25),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            # maxblurpool-k2b3s2=maxpool-k2s1+BlurPool-k3s2
            nn.MaxPool1d(kernel_size=2, stride=1),
            BlurPool(channels=16, filt_size=3, stride=2),

            # Regressor
            nn.Conv1d(in_channels=16, out_channels=3+nbclass, kernel_size=1, stride=1),
        )

    def forward(self, x:torch.Tensor):
        '''
        output shape   (bs, nb_anchors, nl, 3+nbclass)
        '''
        output = self.net_architecture(x)
        output = output.permute([0,2,1]) # (bs, 259, nl) -> (bs, nl, 259)
        output[..., 0] = output[..., 0].sigmoid()  # sigmoid(ps_t)
        output[..., 2] = output[..., 2].sigmoid()  # conf
        return output
    

class ACNN_DPAv4_woblur(nn.Module):
    """ACNN model for DPAv4"""
    def __init__(self, nbclass:int):
        super(ACNN_DPAv4_woblur, self).__init__()
        self.net_architecture = nn.Sequential(
            # Feature blocks
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=51, stride=1, padding=25),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=25, stride=25),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),

            # Regressor
            nn.Conv1d(in_channels=16, out_channels=3+nbclass, kernel_size=1, stride=1),
        )

    def forward(self, x:torch.Tensor):
        '''
        output shape   (bs, nb_anchors, nl, 3+nbclass)
        '''
        output = self.net_architecture(x)
        output = output.permute([0,2,1]) # (bs, 259, nl) -> (bs, nl, 259)
        output[..., 0] = output[..., 0].sigmoid()  # sigmoid(ps_t)
        output[..., 2] = output[..., 2].sigmoid()  # conf
        return output


class ACNN_ASCAD_wocombine(nn.Module):
    """ACNN_DPAv4 for ASCAD dataset"""
    def __init__(self, nbclass:int):
        super(ACNN_ASCAD_wocombine, self).__init__()
        self.net_architecture = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            BlurPool(channels=32, filt_size=3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            BlurPool(channels=32, filt_size=3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=3+nbclass, kernel_size=1, stride=1),
        )

    def forward(self, x:torch.Tensor):
        '''
        input shape: (bs, 1, trace_length)
        output shape: (bs, nl, 3+nbclass)
        '''
        output = self.net_architecture(x) 
        output = output.permute([0,2,1]) # (bs, 259, nl) -> (bs, nl, 259)
        output[..., 0] = output[..., 0].sigmoid()  # sigmoid(ps_t)
        output[..., 2] = output[..., 2].sigmoid()  # conf
        return output


class ACNN_ASCAD(nn.Module):
    """ACNN_DPAv4 for ASCAD dataset"""
    def __init__(self, nbclass:int):
        super(ACNN_ASCAD, self).__init__()
        self.net_architecture = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            BlurPool(channels=32, filt_size=3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            BlurPool(channels=32, filt_size=3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=13, kernel_size=1, stride=1),
        )
        self.convert = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=10, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=10, out_channels=nbclass, kernel_size=1, stride=1)
            )

    def forward(self, x:torch.Tensor):
        '''
        input shape: (bs, 1, trace_length)
        output shape: (bs, nl, 3+nbclass)
        '''
        x = self.net_architecture(x) #(bs, 1, trace_length)->(bs, 13, nl)
        cb_x = x.permute([0,2,1])[...,3:] # (bs, 13, nl) -> (bs, nl, 13) then isolate the feature vector -> (bs, nl, 10)
        cb_x = torch.matmul(cb_x.unsqueeze(-1), cb_x.unsqueeze(-2)) # combine layer: cross product (bs, nl, 10, 10)
        cb_x =  self.convert(cb_x.view(x.shape[0],-1, 100).permute([0,2,1])) # Flatten -> (bs, nl, 100) -> (bs, 100, nl) clss prediction -> (bs, 256, nl)
        output = torch.cat([x[:, :3, :], cb_x], dim=1) # Concat  (bs, 3, nl) + (bs, 256, nl) -> (bs, 259, nl)
        output = output.permute([0,2,1]) # (bs, 259, nl) -> (bs, nl, 259)
        output[..., 0] = output[..., 0].sigmoid()  # sigmoid(ps_t)
        output[..., 2] = output[..., 2].sigmoid()  # conf
        return output


    
def load_model(nbclass:int, model_path=None, name:str='ASCAD'):
    """Loads the model.
    ----------------------
    Input:
    - nbclass: number of classes.
    - model_path: Path to weights or checkpoint file (.pth).
    - name: which network model to load. Option: ASCAD(default) and DPAv4. 

    Output: A model instance.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device for inference
    if name=='ASCAD':
        model = ACNN_ASCAD(nbclass=nbclass).to(device)
    elif name=='ASCAD_wocombine':
        model = ACNN_ASCAD_wocombine(nbclass=nbclass).to(device)
    elif name=='DPAv4':
        model = ACNN_DPAv4(nbclass=nbclass).to(device)
    elif name=='DPAv4_woblur':
        model = ACNN_DPAv4_woblur(nbclass=nbclass).to(device)
    else:
        exit()
    # If pretrained weights are specified, start from checkpoint
    if model_path:
        if model_path.endswith(".pth"):
            # Load checkpoint weights
            f = torch.load(model_path, map_location=device)
            print('Model saved from epoch-{}'.format(f['epoch']))
            model.load_state_dict(f['model_parameter'])
    return model


class BlurPool(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:].repeat((self.channels,1,1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride]
        else:
            # return nn.functional.conv1d(inp, self.filt, stride=self.stride, groups=inp.shape[1])
            return nn.functional.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer