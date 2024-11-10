"""This script prepares two numpy arrays with the data
ready to be used for dataloading for training a model.
"""
import os
import sys
import math
from copy import deepcopy
import argparse

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from esox.fast_io import read_fast5, read_fast
from esox.resquiggle import seq_to_signal
from esox.normalization import med_mad

def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

def regular_break_points(n, chunk_len, overlap=0, align='mid'):
    """Define the start and end points of the raw data based on the 
    window length and overlap
    
    Copied from: https://github.com/nanoporetech/bonito/blob/master/bonito/cli/convert.py
    
    Args:
        n (int): length of the raw data
        chunk_len (int): window size
        overlap (int): overlap between windows
        align (str): relative to the whole length, how should the windows align
    """
    num_chunks, remainder = divmod(n - overlap, chunk_len - overlap)
    start = {'left': 0, 'mid': remainder // 2, 'right': remainder}[align]
    starts = np.arange(start, start + num_chunks*(chunk_len - overlap), (chunk_len - overlap))
    return np.vstack([starts, starts + chunk_len]).T
                
def chunk_read(read_data, window_length, overlap, min_bases, max_bases, min_coverage):
    """Processes a single/multi fast5 read into chunks for deep learning training
    
    Args:
        read_file (str): fast5 file
        window_Length (int): size of the chunks in raw datapoints
        overlap (int): overlap between windows
        min_bases (int): minimum number of bases for a chunk to be considered
        max_bases (int): max number of bases for a chunk to be considered
    """

    read_file = read_data['fast5_file']
    read_id = read_data['read_id']
    med = read_data['med']
    mad = read_data['mad']
    segments = read_data['segments']
    start_rel_to_raw = read_data['start_rel_to_raw']
    reference_seq = read_data['reference_seq']
    reference_seq = reference_seq.upper()
    reference_seq = reference_seq.replace('O', 'o')

    if not os.path.isfile(read_file):
        print('File not found, skipping: ' + read_file)
        return None, None, None
    
    fast5_data = read_fast5(read_file, read_ids = read_id)[read_id]

    _, old_mad = med_mad(fast5_data.raw, factor = 1.0)

    if mad <= 0 or old_mad <= 0:
        return None, None, None
    if np.abs(np.log2(old_mad) - np.log2(mad)) > 1:
        return None, None, None

    sample = (fast5_data.raw - med)/mad
    # segment start and end points
    break_points = regular_break_points(len(fast5_data.raw), window_length, overlap = overlap, align = 'left')

    
    # get the positions of the sequence and the sequence as a whole
    pointers = segments + start_rel_to_raw

    y_long = np.full(sample.shape, '', dtype = 'U')
    y_long[pointers[:-1]] = deepcopy(np.array(list(reference_seq)))
    e_long = np.zeros(sample.shape, dtype = np.float32)
    expected_sig = seq_to_signal(read_data['reference_seq'].replace('o', 'G').upper())
    
    segment_lengths = segments[1:] - segments[:-1]
    long_expected_signal = np.repeat(expected_sig, segment_lengths[2:-3])
    st = pointers[2]
    nd = pointers[-4]
    e_long[st:nd] = long_expected_signal

    x_list = list()
    y_list = list()
    e_list = list()
    for chunk_n, (i, j) in enumerate(break_points):

        y = y_long[i:j]
        basep = np.where(y != '')[0] 
        if len(basep) < min_bases:
            continue
        if len(basep) > max_bases:
            continue

        if chunk_n > 0:
            if basep[-1] - basep[0] < y.shape[0] * min_coverage:
                continue

        # extract the data that we need
        x_list.append(sample[i:j])
        y_list.append(y)
        e_list.append(e_long[i:j])

    if len(x_list) == 0:
        return None, None, None

    # stack all the arrays
    x = np.vstack(x_list)
    y = np.vstack(y_list)
    e = np.vstack(e_list)

    return x, y, e

def chunk_reads_and_write(
    reads_data,
    output_file,
    window_length, 
    overlap, 
    min_bases,
    max_bases,
    min_coverage,
    ):
    """Processes a set of fast5 reads into chunks for deep learning training
    and writes the chunks and labels into a numpy array
    
    Args:
        read_file (str): fast5 file
        output_file (str): numpy file name
        window_Length (int): size of the chunks in raw datapoints
        overlap (int): overlap between windows
        min_bases (int): minimum number of bases for a chunk to be considered
        max_bases (int): max number of bases for a chunk to be considered
        min_coverage (float): minimum fraction of coverage of the basecalls 
        relative to the raw data based on their segmentation.
    """
    
    x_list = list()
    y_list = list()
    e_list = list()
    
    for read_data in tqdm(reads_data):
        x, y, e = chunk_read(read_data, window_length, overlap, min_bases, max_bases, min_coverage)
        if x is None:
            continue
        x_list.append(x)
        y_list.append(y)
        e_list.append(e)
        
    X = np.vstack(x_list)
    Y = np.vstack(y_list)
    E = np.vstack(e_list)

    assert X.shape == Y.shape
    assert X.shape == Y.shape
    
    np.savez(output_file, x = X, y = Y, e = E)

def main(reference_file, fast5_file, resquiggle_file, output_file, 
         window_length, window_overlap, min_bases, max_bases, min_coverage):    

    print('Finding files to process')
    # find all the files that we have to process

    references = read_fast(reference_file)
    resquiggle = read_fast(resquiggle_file)

    all_read_data = list()    
    for read_info, segments in resquiggle.items():
        
        read_data = dict()
        read_data['segments'] = np.array(segments.split(',')).astype(int)
        read_info = read_info.split(';')
        read_data['read_id'] = read_info[0]
        read_data['start_rel_to_raw'] = int(read_info[1])
        read_data['med'] = float(read_info[2])
        read_data['mad'] = float(read_info[3])

        read_data['fast5_file'] = fast5_file
        read_data['reference_seq'] = references[read_data['read_id']][2:-3]
        all_read_data.append(read_data)
        
    chunk_reads_and_write(
        reads_data=all_read_data,
        output_file=output_file,
        window_length=window_length,
        overlap=window_overlap,
        min_bases=min_bases,
        max_bases=max_bases,
        min_coverage=min_coverage,
    )
    
    
    return None

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-file", type=str, help='Path to fast5 files')
    parser.add_argument("--fast5-file", type=str,  help='Path to file with list of files to be processed')
    parser.add_argument("--resquiggle-file", type=str, help='Path to save the numpy arrays')
    parser.add_argument("--output-file", type=str, help='Path to save the numpy arrays')
    
    parser.add_argument("--window-size", type=int, help='Size of the window of a raw data segment', default = 2000)
    parser.add_argument("--window-slide", type=int, help='Number of datapoints of overlap between sequential segments', default = 0)
    parser.add_argument("--min-bases", type=int, help='Minimum number of bases for a segment to be kept', default = 10)
    parser.add_argument("--max-bases", type=int, help='Maximum number of bases for a segment to be kept', default = math.inf)
    parser.add_argument("--min-coverage", type=float, help = 'Minimum fraction of coverage of the basecalls relative to the raw data based on their segmentation', default = 0.8)
    args = parser.parse_args()
    
    main(reference_file = args.reference_file, 
         fast5_file = args.fast5_file, 
         resquiggle_file = args.resquiggle_file,
         output_file = args.output_file,
         window_length = args.window_size, 
         window_overlap = args.window_slide, 
         min_bases = args.min_bases, 
         max_bases = args.max_bases, 
         min_coverage = args.min_coverage)

