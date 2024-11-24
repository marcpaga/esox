from abc import abstractmethod

import torch
from torch import nn

from esox.constants import CRF_STATE_LEN

try:
    from fast_ctc_decode import crf_greedy_search, crf_beam_search
except ImportError:
    pass

try:
    from esox.layers import CTC_CRF
except ImportError:
    pass

class BaseModel(nn.Module):

    def __init__(
        self, 
        device, 
        dataset = None,
        clipping_value = 2, 
        prefetch_factor = None,
    ):
        super(BaseModel, self).__init__()

        self.device = device
        self.dataset = dataset

        # optimization
        self.clipping_value = clipping_value
        self.prefetch_factor = prefetch_factor
        
        self.save_dict = dict()
        self.init_weights()

    @abstractmethod
    def forward(self, batch):
        """Forward through the network
        """
        raise NotImplementedError()
    
    def _process_batch(self, batch):
        return batch
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            batch = self._process_batch(batch)
            x = batch['x'].to(torch.float).to(self.device)
            x = x.unsqueeze(1)
            p = self.forward(x)
            
        return p
    
    def init_weights(self):
        """Initialize weights from uniform distribution
        """
        for _, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        """Count trainable parameters in model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class BaseModelCRF(BaseModel):
    
    def __init__(self, alphabet = 'NACGT', state_len = CRF_STATE_LEN, *args, **kwargs):
        """
        Args:
            state_len (int): k-mer length for the states
            alphabet (str): bases available for states, defaults 'NACGT'
        """
        super(BaseModelCRF, self).__init__(*args, **kwargs)

        self.alphabet = alphabet
        self.n_base = len(self.alphabet) - 1
        self.state_len = state_len
        self.seqdist = CTC_CRF(state_len = self.state_len, alphabet = self.alphabet)

    def _process_batch(self, batch):
        return batch

    def decode(self, p, greedy = True, *args, **kwargs):
        """Decode the predictions
        
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if greedy:
            return self.decode_crf_greedy(p, *args, **kwargs)
        else:
            return self.decode_crf_beamsearch(p, *args, **kwargs)

    def compute_scores(self, probs, use_fastctc = False):
        """
        Args:
            probs (cuda tensor): [length, batch, channels]
            use_fastctc (bool)
        """
        if use_fastctc:
            scores = probs.cuda().to(torch.float32)
            betas = self.seqdist.backward_scores(scores.to(torch.float32))
            trans, init = self.seqdist.compute_transition_probs(scores, betas)
            trans = trans.to(torch.float32).transpose(0, 1)
            init = init.to(torch.float32).unsqueeze(1)
            return (trans, init)
        else:
            scores = self.seqdist.posteriors(probs.cuda().to(torch.float32)) + 1e-8
            tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
            return tracebacks

    def _decode_crf_greedy_fastctc(self, tracebacks, init, qstring, qscale, qbias, return_path):
        """
        Args:
            tracebacks (np.array): [len, states, bases]
            init (np.array): [states]
            qstring (bool)
            qscale (float)
            qbias (float)
            return_path (bool)
        """

        seq, path = crf_greedy_search(
            network_output = tracebacks, 
            init_state = init, 
            alphabet = self.alphabet, 
            qstring = qstring, 
            qscale = qscale, 
            qbias = qbias
        )
        if return_path:
            return seq, path
        else:
            return seq
    
    def decode_crf_greedy(self, probs, use_fastctc = False, qstring = False, qscale = 1.0, qbias = 1.0, return_path = False, *args, **kwargs):
        """Predict the sequences using a greedy approach
        
        Args:
            probs (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """

        if use_fastctc:
            tracebacks, init = self.compute_scores(probs, use_fastctc)
            return self._decode_crf_greedy_fastctc(tracebacks.cpu().numpy(), init.cpu().numpy(), qstring, qscale, qbias, return_path)
        
        else:
            tracebacks = self.compute_scores(probs, use_fastctc).cpu().numpy()
            if return_path:
                return [self.seqdist.path_to_str(y) for y in tracebacks], tracebacks
            else:
                return [self.seqdist.path_to_str(y) for y in tracebacks]

    def _decode_crf_beamsearch_fastctc(self, tracebacks, init, beam_size, beam_cut_threshold, return_path):
        """
        Args
            tracebacks (np.array): [len, states, bases]
            init (np.array): [states]
            beam_size (int)
            beam_cut_threshold (float)
            return_path (bool)
        """
        seq, path = crf_beam_search(
            network_output = tracebacks, 
            init_state = init, 
            alphabet = self.alphabet, 
            beam_size = beam_size,
            beam_cut_threshold = beam_cut_threshold
        )
        if return_path:
            return seq, path
        else:
            return seq

    def decode_crf_beamsearch(self, probs, beam_size = 5, beam_cut_threshold = 0.1, return_path = False, *args, **kwargs):
        """Predict the sequences using a beam search
        
        Args:
            probs (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """

        tracebacks, init = self.compute_scores(probs, use_fastctc = True)
        return self._decode_crf_beamsearch_fastctc(tracebacks, init, beam_size, beam_cut_threshold, return_path)


class BaseRemoraModel(BaseModel):
    
    def __init__(self, *args, **kwargs):
        """
        Args:
            state_len (int): k-mer length for the states
            alphabet (str): bases available for states, defaults 'NACGT'
        """
        super(BaseRemoraModel, self).__init__(*args, **kwargs)

    def _process_batch(self, batch):

        for k, v in batch.items():
            if k in ['x', 'e1', 'd1', 'e2', 'd2', 'ce', 'cd']:
                v = v.to(self.device).to(torch.float)
                if len(v.shape) == 2:
                    v = v.unsqueeze(1)
                batch[k] = v
            if k in ['s1', 'p1', 's2', 'p2', 'cs', 'cp']:
                batch[k] = v.to(self.device).to(torch.int)

        return batch
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            batch = self._process_batch(batch)
            p, _ = self.forward(batch)
            
        return p.exp()
    
    def train_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """

        batch = self._process_batch(batch)
        logits, _ = self.forward(batch)
            
        return logits
    
    def validation_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """

        self.eval()
        with torch.no_grad():
            batch = self._process_batch(batch)
            logits, _ = self.forward(batch)
            
        return logits

 