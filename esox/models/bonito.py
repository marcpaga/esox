import os
import sys

from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from esox.modelclasses import BaseModelCRF
from esox.layers import BonitoLinearCRFDecoder, BonitoLSTM
from esox.constants import CRF_STATE_LEN, CRF_BIAS, CRF_SCALE, CRF_BLANK_SCORE

class BonitoModel(BaseModelCRF):
    """Bonito Model
    """
    def __init__(self, size, *args, **kwargs):
        super(BonitoModel, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            reverse (bool): if the first rnn layer starts with reverse 
        """
    
        self.size = size
            
        self.cnn = self.build_cnn()
        self.enc = self.build_encoder()
        self.dec = self.build_decoder()
        
        self.stride = 5

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        
        x = self.cnn(x)
        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.enc(x)
        x = self.dec(x)
        return x

    def build_cnn(self):

        cnn = nn.Sequential(
            nn.Conv1d(
                in_channels = 1, 
                out_channels = 4, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = 4, 
                out_channels = 16, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = 16, 
                out_channels = self.size, 
                kernel_size = 19, 
                stride= 5, 
                padding=19//2, 
                bias=True),
            nn.SiLU()
        )
        return cnn

    def build_encoder(self, reverse = True):

        if reverse:
            encoder = nn.Sequential(BonitoLSTM(self.size, self.size, reverse = True),
                                    BonitoLSTM(self.size, self.size, reverse = False),
                                    BonitoLSTM(self.size, self.size, reverse = True),
                                    BonitoLSTM(self.size, self.size, reverse = False),
                                    BonitoLSTM(self.size, self.size, reverse = True))
        else:
            encoder = nn.Sequential(BonitoLSTM(self.size, self.size, reverse = False),
                                    BonitoLSTM(self.size, self.size, reverse = True),
                                    BonitoLSTM(self.size, self.size, reverse = False),
                                    BonitoLSTM(self.size, self.size, reverse = True),
                                    BonitoLSTM(self.size, self.size, reverse = False))
        return encoder    

        
    def build_decoder(self):

        decoder = BonitoLinearCRFDecoder(
                insize = self.size, 
                n_base = self.n_base, 
                state_len = CRF_STATE_LEN, 
                bias=CRF_BIAS, 
                scale= CRF_SCALE, 
                blank_score= CRF_BLANK_SCORE
            )
        return decoder

