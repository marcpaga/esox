# Functions to prepare and deal with Remora style inputs
import re
from copy import deepcopy

import torch
import numpy as np
from numba import jit
from tombo.tombo_helper import TomboError

from esox.resquiggle import resquiggle_read_normalized, seq_to_signal
from esox.constants import (
    ACC_GLOBAL_ALIGN_FUNCTION, 
    GAP_OPEN_PENALTY, 
    GAP_EXTEND_PENALTY, 
    ENCODING_DICT_CRF,
    PHRED_LETTERS
)

def elongate_cigar(short_cigar):
    cigar_counts = re.split('H|X|=|I|D|N|S|P|M', short_cigar)
    cigar_strs = re.split('[0-9]', short_cigar)
    
    cigar_counts = [c for c in cigar_counts if c != '']
    cigar_strs = [c for c in cigar_strs if c != '']
    
    assert len(cigar_strs) == len(cigar_counts)
    
    longcigar = ''
    for c, s in zip(cigar_counts, cigar_strs):
        longcigar += s*int(c)
    return longcigar, cigar_counts, cigar_strs

def stitch_by_stride(chunks, chunksize, overlap, length, stride):
    """
    Stitch chunks together with a given overlap
    
    This works by calculating what the overlap should be between two outputed
    chunks from the network based on the stride and overlap of the inital chunks.
    The overlap section is divided in half and the outer parts of the overlap
    are discarded and the chunks are concatenated. There is no alignment.
    
    Chunk1: AAAAAAAAAAAAAABBBBBCCCCC
    Chunk2:               DDDDDEEEEEFFFFFFFFFFFFFF
    Result: AAAAAAAAAAAAAABBBBBEEEEEFFFFFFFFFFFFFF
    
    Args:
        chunks (tensor): predictions with shape [samples, length, *]
        chunk_size (int): initial size of the chunks
        overlap (int): initial overlap of the chunks
        length (int): original length of the signal
        stride (int): stride of the model
        reverse (bool): if the chunks are in reverse order
        
    Copied from https://github.com/nanoporetech/bonito
    """

    if isinstance(chunks, np.ndarray):
        chunks = torch.from_numpy(chunks)

    if chunks.shape[0] == 1: return chunks.squeeze(0)

    semi_overlap = overlap // 2
    start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
    stub = (length - overlap) % (chunksize - overlap)
    first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end


    return torch.cat([
        chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
    ])

def bonito_basecall(x, model, batch_size, orig_len = None, stride = 5, chunksize = 2000, overlap = 400, whole_read = False):
    """
    Args:
        x (array): with chunked raw data
        model: bonito model
        batch_size (int): if the batch size of x is larger than specified batch size,
        do it in steps
        orig_len (int): original length of the whole read
        stride (int): stride of the model
        chunk_size (int): size of the chunks
        overlap (int): overlap between chunks
        whole_read (bool): if x contains a whole read, and therefore should be
        stiched together
    """

    l = x.shape[0]
    ss = torch.arange(0, l, batch_size)
    nn = ss + batch_size

    preds = list()
    for s, n in zip(ss, nn):
        p = model.predict_step({'x':x[s:n, :]})
        preds.append(p)
    if len(preds) > 1:
        preds = torch.cat(preds, dim = 1)
    else:
        preds = preds[0]
    
    if whole_read:
        preds = stitch_by_stride(
            preds.permute(1, 0, 2), 
            chunksize,
            overlap, 
            orig_len, 
            stride,
        )
        # add batch dim since stitch_by_stride output is [len, channels]
        preds = preds.unsqueeze(0).permute(1, 0, 2) # permute for compute_scores to -> [len, batch, channels]

    tracebacks, init = model.compute_scores(preds, use_fastctc = True)

    all_basecalls = list()
    all_phredq = list()
    all_path = list()
    for i in range(preds.shape[1]):
        seq, path = model._decode_crf_greedy_fastctc(
            tracebacks[i].cpu().numpy(), 
            init[i].squeeze(0).cpu().numpy(), 
            qstring = True, 
            qscale = 1, 
            qbias = 0, 
            return_path = True
        )

        basecalls = seq[:len(path)]
        phredq = np.array(list(seq[len(path):]))
        all_basecalls.append(basecalls)
        all_phredq.append(phredq)
        all_path.append(np.array(path) * stride)

    if whole_read:
        return all_basecalls[0], all_phredq[0], all_path[0]
    else:
        return all_basecalls, all_phredq, all_path


def make_align_arr_int(long_cigar, truth_seq, pred_seq, path_idx):
    """Makes an alignment array based on the long cigar
    
    Args:
        long_cigar (str): output from `elongate_cigar`
        truth_seq (np.ndarray[int]): truth seq as a int numpy array
        pred_seq (np.ndarray[int]): pred seq as a int numpy array
        path_idx (np.ndarray[int]): index path to original size of predicitions
    Returns:
        A np:array of shape [4, alignment_length]. 
    """
    
    tc = 0
    pc = 0
    align_arr = np.zeros((4, len(long_cigar)), dtype = np.int64)
    for i, c in enumerate(long_cigar):
        if c == 'D' or c == 'N':
            align_arr[0, i] = truth_seq[tc]
            align_arr[1, i] = -1
            align_arr[2, i] = -1
            align_arr[3, i] = -1

            tc += 1
        elif c == 'I':
            align_arr[0, i] = -1
            align_arr[1, i] = 2
            align_arr[2, i] = pred_seq[pc]
            align_arr[3, i] = path_idx[pc]

            pc += 1
        elif c == 'X':
            align_arr[0, i] = truth_seq[tc]
            align_arr[1, i] = 0
            align_arr[2, i] = pred_seq[pc]
            align_arr[3, i] = path_idx[pc]

            pc += 1
            tc += 1
        elif c == '=':
            align_arr[0, i] = truth_seq[tc]
            align_arr[1, i] = 1
            align_arr[2, i] = pred_seq[pc]
            align_arr[3, i] = path_idx[pc]

            pc += 1
            tc += 1
        elif c == 'M':
            align_arr[0, i] = truth_seq[tc]
            align_arr[2, i] = pred_seq[pc]
            if truth_seq[tc] == pred_seq[pc]:
                align_arr[1, i] = 1
            else:
                align_arr[1, i] = 0
            align_arr[3, i] = path_idx[pc]

            pc += 1
            tc += 1
            
    return align_arr

def bonito_output_to_resquiggled_mapping(x, basecalls, phredq):
    """
    x (np.ndarray): raw data in 1D, shape = [len]
    basecalls (str): string with the basecalls
    phredq (str): string with the phredq scores    
    """

    elongated_path = np.zeros(x.shape, dtype = np.int8)
    elongated_expected = np.zeros(x.shape, dtype = x.dtype)
    elongated_phredq = np.zeros(x.shape, dtype = np.int8)

    try:
        resq_res = resquiggle_read_normalized(
            read_id = 'a', 
            raw_signal = x, 
            genome_seq = basecalls, 
            norm_signal = x,
        )
    except TomboError:
        return elongated_path, elongated_phredq, elongated_expected

    encoded_basecalls = np.vectorize(ENCODING_DICT_CRF.get)(np.array(list(basecalls)))
    encoded_phredq = np.vectorize(PHRED_LETTERS.get)(phredq)

    pointers = resq_res.segs + resq_res.read_start_rel_to_raw

    elongated_path[pointers] = encoded_basecalls[2:-2]
    elongated_phredq[pointers] = encoded_phredq[2:-2]

    expected_signal = np.repeat(seq_to_signal(basecalls), resq_res.segs[1:] - resq_res.segs[:-1])
    elongated_expected[pointers[0]:pointers[-1]] = expected_signal

    return elongated_path, elongated_phredq, elongated_expected
    

@jit(nopython=True)
def find_cut_points_jit(g_calls, original_len, half_window, cuts, cuts_with_coords):
    """Given a list of positions of G calls and the calls path, return the list
    of cutting points for the raw data and the output data of a Bonito model.

    Args:
        g_calls (np.ndarray): list of positions where a G is called in path_long
        path_long (np.ndarray): path with the calls, has to be output length
        with zeros as filling.
        orig_idx (np.ndarray): array that indicates the index of the original
        input to bonito, for the output array based on the stride of the conv.
        original_len (int): initial length of the input to bonito
        half_window (int): half size of the window for remora
        cuts (np.ndarray): placeholder zeros array for the cut positions, has to be
        of shape (len(g_calls), 4)
        cuts_with_coords (np.ndarray): placeholder zeros bool array to subset
        the cuts array to only positions with cuts.

    Returns a  np.ndarray with shape (cut positions, 4), the first two positions
    correspond to the raw data (bonito input shape), the last two positions
    correspond to the output of bonito shape.

    """

    for i, pos_idx in enumerate(g_calls):
        
        st = pos_idx - half_window
        if st < 0:
            continue
        nd = pos_idx + half_window
        if nd >= original_len:
            continue
        
        cuts[i] = [st, nd]
        cuts_with_coords[i] = True
        

    return cuts[cuts_with_coords], cuts_with_coords

@jit(nopython=True)
def cut_data_jit(cuts, x, x_cut_array_placeholder):
    """Cuts the raw data based on the cuts array

    Args:
        cuts (np.ndarray) array with cut positions, 2D with shape (cuts, (st, nd))
        x (np.ndarray): array with raw data to be cut
        x_cut_array_placeholder (np.ndarray): placeholder to put the raw data,
        has shape (cuts, cut_window)
    """

    for i, cut in enumerate(cuts):

        st, nd = cut[0], cut[1]
        x_cut_array_placeholder[i] = x[st:nd]

    return x_cut_array_placeholder

def get_mod_pos(truth, basecall, path, min_acc = 0.8, unmod_base = 3, mod_base = 5, encode_dict = ENCODING_DICT_CRF):

    align_arr, long_cigar = align(deepcopy(truth), basecall, path, encode_dict = encode_dict)

    # no alignment skip
    if long_cigar is None:
        return list(), list(), list()

    # accuracy too low skip
    acc = np.sum(align_arr[1] == 1)/len(long_cigar)

    if acc < min_acc:
        return list(), list(), list()

    return getmod_pos_dist_label(align_arr, unmod_base = unmod_base, mod_base = mod_base)

@jit(nopython=True)
def getmod_pos_dist_label(align_arr, unmod_base = 3, mod_base = 5):
    '''Given an align arr, find the guanine calls positions, if they are 8oxog,
    distance to 8oxog and label
    Labels:
        - 1 -> 8-oxoG
        - 0 -> G
        - -1 -> G basecall, but not G reference

    Distance is just number of bases between an 8oxog and the G call, if there's
    no 8-oxoG then it writes -1.
    '''

    pos_idxs = list()
    dist_to_mod = list()
    label = list()
    mod_pos = np.where(align_arr[0] == mod_base)[0]

    for i, (r, c, b, p) in enumerate(align_arr.transpose()):
        # r = reference
        # c = cigar alignment
        # b = basecall
        # p = path position

        # skip things at start and end of the alignment
        if i < 2 or i > align_arr.shape[1] - 3:
            continue

        # if not basecall G, continue
        if b != unmod_base:
            continue
        
        # if we basecall a G
        # if reference is a oxoG and no errors around 2 bases
        if r == mod_base and align_arr[1, i-2:i+3].sum() == 5:
            pos_idxs.append(p)
            dist_to_mod.append(0)
            label.append(1)
            continue


        pos_idxs.append(p)
        if len(mod_pos) > 0:
            dist_to_mod.append(
                np.abs(mod_pos - i).min()
            )
        else:
            dist_to_mod.append(-1)

        # label according if label is G or not
        if r == unmod_base:
            label.append(0)
        else:
            label.append(-1)

    return np.array(pos_idxs), np.array(dist_to_mod), np.array(label)


def mask_basecalls(calls, non_masked_bases, masking_value):

    b_pos = np.where(calls != masking_value)[0]
    m_pos = np.searchsorted(b_pos, calls.shape[0]/2).item()

    if len(b_pos[:m_pos]) > non_masked_bases:
        calls[:b_pos[m_pos-non_masked_bases]] = masking_value

    if len(b_pos[m_pos+1:]) > non_masked_bases:
        calls[b_pos[m_pos+1+non_masked_bases]:] = masking_value

    return calls


def prepare_remora_input_for_predict(norm_raw_data, paths, stub, stride, cut_window = 50, target_base = 3):

    start_window_length = len(norm_raw_data)
    orig_idx = np.arange(0, start_window_length-stub, stride, dtype = int)
    orig_idx_st = orig_idx - cut_window
    orig_idx_nd = orig_idx + cut_window

    assert len(orig_idx) == len(paths)

    originalx = norm_raw_data[stub:]
    originalp = np.zeros((4, originalx.shape[0]), dtype=float)

    valid_idxs = np.ones(paths.shape, dtype=bool)
    valid_idxs[paths != target_base] = False
    valid_idxs[orig_idx_st < 0] = False
    valid_idxs[orig_idx_nd >= len(originalx)] = False
    valid_idx_pos = np.where(valid_idxs)[0]

    base_pos = np.where(paths != 0)[0]

    cut_x = np.zeros((len(valid_idx_pos), cut_window*2), dtype=float)
    cut_seq = np.zeros((len(valid_idx_pos), 4, cut_window*2), dtype=float)

    originalp[paths[base_pos]-1, orig_idx[base_pos]] = 1

    for i, (s, n) in enumerate(zip(orig_idx_st[valid_idx_pos], orig_idx_nd[valid_idx_pos])):
        cut_x[i, :] = originalx[s:n]
        cut_seq[i, :, :] = originalp[:, s:n]

    return cut_x, cut_seq, valid_idxs