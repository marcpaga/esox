import os
import sys
import warnings

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from esox.constants import CRF_STATE_LEN, CRF_BIAS, CRF_SCALE, CRF_BLANK_SCORE

class BonitoLinearCRFDecoder(nn.Module):
    """Linear decoder that has scores instead of bases to have
    base-base transitions to be able to use a CRF
    
    Copied from https://github.com/nanoporetech/bonito
    """

    def __init__(self, insize, n_base = 4, state_len = CRF_STATE_LEN, bias=CRF_BIAS, scale= CRF_SCALE, blank_score= CRF_BLANK_SCORE):
        super(BonitoLinearCRFDecoder, self).__init__()
        """
        Args:
            insize (int): number of input channels
            n_base (int): number of bases
            state_len (int):
            bias (bool): whether the linear layer has bias
            scale (float): scores are multiplied by this value after tanh activation
            blank_score (float): 
            
        Defaults are based on the guppy configuration file
        """
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
        self.linear = nn.Linear(insize, size, bias=bias)
        self.activation = nn.Tanh()
        self.scale = scale
        
    def forward(self, x):
        """
        Args:
            x (tensor): tensor with shape (timesteps, batch, channels)
        """
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None:
            T, N, C = scores.shape
            s = torch.tensor(self.blank_score, device=scores.device, dtype=scores.dtype)
            scores = torch.cat([s.expand(T, N, C//self.n_base, 1), scores.reshape(T, N, C//self.n_base, self.n_base)], 
                              axis=-1).reshape(T, N, -1)
            # T, N, C = scores.shape
            # scores = torch.nn.functional.pad(
            #     scores.view(T, N, C // self.n_base, self.n_base),
            #     (1, 0, 0, 0, 0, 0, 0, 0),
            #     value=self.blank_score
            # ).view(T, N, -1)
        return scores
    
try:
    import seqdist.sparse
    from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
    from seqdist.core import SequenceDist, Max, Log, semiring
    
    class CTC_CRF(SequenceDist):
        """CTC-CRF mix
        Copied from https://github.com/nanoporetech/bonito
        """

        def __init__(self, state_len, alphabet):
            super().__init__()
            self.alphabet = alphabet
            self.state_len = state_len
            self.n_base = len(alphabet[1:])
            self.idx = torch.cat([
                torch.arange(self.n_base**(self.state_len))[:, None],
                torch.arange(
                    self.n_base**(self.state_len)
                ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
            ], dim=1).to(torch.int32)

            self.scores_dim = (self.n_base + 1) * self.n_base**self.state_len

        def n_score(self):
            return len(self.alphabet) * self.n_base**(self.state_len)

        def logZ(self, scores, S:semiring=Log):
            T, N, _ = scores.shape
            Ms = scores.reshape(T, N, -1, len(self.alphabet))
            alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
            beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
            return seqdist.sparse.logZ(Ms, self.idx, alpha_0, beta_T, S)

        def normalise(self, scores):
            return (scores - self.logZ(scores)[:, None] / len(scores))

        def forward_scores(self, scores, S: semiring=Log):
            T, N, _ = scores.shape
            Ms = scores.reshape(T, N, -1, self.n_base + 1)
            alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
            return seqdist.sparse.fwd_scores_cupy(Ms, self.idx, alpha_0, S, K=1)

        def backward_scores(self, scores, S: semiring=Log):
            T, N, _ = scores.shape
            Ms = scores.reshape(T, N, -1, self.n_base + 1)
            beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
            return seqdist.sparse.bwd_scores_cupy(Ms, self.idx, beta_T, S, K=1)

        def compute_transition_probs(self, scores, betas):
            T, N, C = scores.shape
            # add bwd scores to edge scores
            log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
            # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
            log_trans_probs = torch.cat([
                log_trans_probs[:, :, :, [0]],
                log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
            ], dim=-1)
            # convert from log probs to probs by exponentiating and normalising
            trans_probs = torch.softmax(log_trans_probs, dim=-1)
            #convert first bwd score to initial state probabilities
            init_state_probs = torch.softmax(betas[0], dim=-1)
            return trans_probs, init_state_probs

        def reverse_complement(self, scores):
            T, N, C = scores.shape
            expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
            scores = scores.reshape(*expand_dims)
            blanks = torch.flip(scores[..., 0].permute(
                0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
            )
            emissions = torch.flip(scores[..., 1:].permute(
                0, 1, *range(self.state_len, 1, -1),
                self.state_len +2,
                self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
            )
            return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

        def viterbi(self, scores):
            traceback = self.posteriors(scores, Max)
            paths = traceback.argmax(2) % len(self.alphabet)
            return paths

        def path_to_str(self, path):
            alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
            seq = alphabet[path[path != 0]]
            return seq.tobytes().decode()

        def prepare_ctc_scores(self, scores, targets):
            # convert from CTC targets (with blank=0) to zero indexed
            targets = torch.clamp(targets - 1, 0)

            T, N, C = scores.shape
            scores = scores.to(torch.float32)
            n = targets.size(1) - (self.state_len - 1)
            stay_indices = sum(
                targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
                for i in range(self.state_len)
            ) * len(self.alphabet)
            move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
            stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
            move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
            return stay_scores, move_scores

        def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
            if normalise_scores:
                scores = self.normalise(scores)
            stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
            logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
            loss = - (logz / target_lengths)
            if loss_clip:
                loss = torch.clamp(loss, 0.0, loss_clip)
            if reduction == 'mean':
                return loss.mean()
            elif reduction in ('none', None):
                return loss
            else:
                raise ValueError('Unknown reduction type {}'.format(reduction))

        def ctc_viterbi_alignments(self, scores, targets, target_lengths):
            stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
            return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)

except ImportError as e:
    warnings.warn(str(e))    
    warnings.warn('CTC-CRF Decoder not available')

    
class BonitoLSTM(nn.Module):
    """Single LSTM RNN layer that can be reversed.
    Useful to stack forward and reverse layers one after the other.
    The default in pytorch is to have the forward and reverse in
    parallel.
    
    Copied from https://github.com/nanoporetech/bonito
    """
    def __init__(self, in_channels, out_channels, reverse = False):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            reverse (bool): whether to the rnn direction is reversed
        """
        super(BonitoLSTM, self).__init__()
        
        self.rnn = nn.LSTM(in_channels, out_channels, num_layers = 1, bidirectional = False, bias = True)
        self.reverse = reverse
        
    def forward(self, x):
        if self.reverse: x = x.flip(0)
        y, h = self.rnn(x)
        if self.reverse: y = y.flip(0)
        return y

class RemoraSignalCNN(nn.Module):
    """Convolution on the signal input of a Remora model
    """

    def __init__(self, size, dropout, *args, **kwargs):
        super(RemoraSignalCNN, self).__init__(*args, **kwargs)

        self.size = size

        self.sig_conv1 = nn.Conv1d(1, 4, 5)
        self.sig_conv2 = nn.Conv1d(4, 16, 5)
        self.sig_conv3 = nn.Conv1d(16, size, 9, 3)

        self.sig_bn1 = nn.BatchNorm1d(4)
        self.sig_bn2 = nn.BatchNorm1d(16)
        self.sig_bn3 = nn.BatchNorm1d(size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        x = F.silu(self.sig_bn1(self.sig_conv1(x)))
        x = F.silu(self.sig_bn2(self.sig_conv2(self.dropout(x))))
        x = F.silu(self.sig_bn3(self.sig_conv3(self.dropout(x))))
        return x


class RemoraSeqCNN(nn.Module):
    """Convolution on the sequence input of a Remora model
    """

    def __init__(self, input_size, emb_size, cnn_size, dropout = 0.1, *args, **kwargs):
        super(RemoraSeqCNN, self).__init__(*args, **kwargs)

        self.size = cnn_size

        self.emb = nn.Embedding(input_size, embedding_dim = emb_size)
        self.seq_conv1 = nn.Conv1d(emb_size, 16, 5)
        self.seq_conv2 = nn.Conv1d(16, cnn_size, 13, 3)
        self.seq_bn1 = nn.BatchNorm1d(16)
        self.seq_bn2 = nn.BatchNorm1d(cnn_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        input:  [b, l]
        output: [b, h, l]
        """
        x = self.emb(x).permute(0, 2, 1)

        x = F.silu(self.seq_bn1(self.seq_conv1(x)))
        x = F.silu(self.seq_bn2(self.seq_conv2(self.dropout(x))))
        return x


class RemoraCRFCNN(nn.Module):
    """Convolution on the CRF input of a Remora model
    """

    def __init__(self, size, crf_input_size = 1280, dropout = 0.1, *args, **kwargs):
        super(RemoraCRFCNN, self).__init__(*args, **kwargs)

        self.size = size
        self.crf_input_size = crf_input_size

        self.seq_conv1 = nn.Conv1d(self.crf_input_size, 512, kernel_size = 1, stride = 1)
        self.seq_conv2 = nn.Conv1d(512, 256, kernel_size = 5, stride = 1)
        self.seq_conv3 = nn.Conv1d(256, size, kernel_size = 13, stride = 3)
        self.seq_bn1 = nn.BatchNorm1d(512)
        self.seq_bn2 = nn.BatchNorm1d(256)
        self.seq_bn3 = nn.BatchNorm1d(size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = F.silu(self.seq_bn1(self.seq_conv1(x)))
        x = F.silu(self.seq_bn2(self.seq_conv2(self.dropout(x))))
        x = F.silu(self.seq_bn3(self.seq_conv3(self.dropout(x))))
        return x

class RemoraLSTM(nn.Module):

    def __init__(self, input_size, size, dropout = 0.5, *args, **kwargs):
        super(RemoraLSTM, self).__init__(*args, **kwargs)

        self.dropout = nn.Dropout(p=dropout)
        self.merge_conv1 = nn.Conv1d(input_size, size, 5)
        self.merge_bn = nn.BatchNorm1d(size)

        self.lstm1 = nn.LSTM(size, size, 1)
        self.lstm2 = nn.LSTM(size, size, 1)
        self.fc = nn.Linear(size, 2)

    def forward(self, x):

        x = F.silu(self.merge_bn(self.merge_conv1(x)))
        x = x.permute(2, 0, 1)
        x = F.silu(self.lstm1(self.dropout(x))[0])
        x = torch.flip(F.silu(self.lstm2(torch.flip(self.dropout(x), (0,)))[0]), (0,))
        emb = x[-1].permute(0, 1)

        x = self.fc(self.dropout(emb))
        return x, emb
