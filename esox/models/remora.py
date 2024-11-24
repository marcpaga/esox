import os
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from esox.modelclasses import BaseRemoraModel
from esox.layers import RemoraSignalCNN, RemoraSeqCNN, RemoraLSTM

class RemoraModel(BaseRemoraModel):
    """Classic Remora Model
    """
    def __init__(
            self, 
            size,
            dropout,
            use_raw: Optional[bool] = True,
            use_seq_1: Optional[bool] = True,
            use_expected_signal_1: Optional[bool] = False,
            use_differential_signal_1: Optional[bool] = False,
            use_phredq_1: Optional[bool] = False,
            use_seq_2: Optional[bool] = False,
            use_expected_signal_2: Optional[bool] = False,
            use_differential_signal_2: Optional[bool] = False,
            use_phredq_2: Optional[bool] = False,
            use_comparative_expected: Optional[bool] = False,
            use_comparative_differential: Optional[bool] = False,
            use_comparative_seq: Optional[bool] = False,
            use_comparative_phredq: Optional[bool] = False,
            *args, 
            **kwargs,
        ):
        super(RemoraModel, self).__init__(*args, **kwargs)
        """
        Args:
        """

        self.params = {
            'sig': {
                'x': use_raw,
                'e1': use_expected_signal_1,
                'd1': use_differential_signal_1,
                'e2': use_expected_signal_2,
                'd2': use_differential_signal_2,
                'ce': use_comparative_expected,
                'cd': use_comparative_differential,
            },
            'seq': {
                's1': (use_seq_1, 5, 8),
                's2': (use_seq_2, 5, 8),
                'p1': (use_phredq_1, 100, 16),
                'p2': (use_phredq_2, 100, 16),
                'cs': (use_comparative_seq, 10, 16),
                'cp': (use_comparative_phredq, 200, 32),
            }
        }

        self.sig_layers = torch.nn.ModuleDict()
        for k, v in self.params['sig'].items():
            if v:
                self.sig_layers[k] = RemoraSignalCNN(size, dropout = dropout)

        self.seq_layers = torch.nn.ModuleDict()
        for k, v in self.params['seq'].items():
            if v[0]:
                self.seq_layers[k] = RemoraSeqCNN(
                    input_size = v[1], 
                    emb_size = v[2], 
                    cnn_size = size, 
                    dropout = dropout
                )

        self.lstm = RemoraLSTM(
            (len(self.sig_layers) + len(self.seq_layers))*size, 
            size, 
            dropout = dropout
        )
    
    def forward(self, batch):

        sig_embeddings = list()
        for k, layer in self.sig_layers.items():
            sig_embeddings.append(layer(batch[k]))
        sig_embeddings = torch.cat(sig_embeddings, dim = 1)

        seq_embeddings = list()
        for k, layer in self.seq_layers.items():
            seq_embeddings.append(layer(batch[k]))
        seq_embeddings = torch.cat(seq_embeddings, dim = 1)

        return self.lstm(torch.cat((sig_embeddings, seq_embeddings), 1))
    
