from copy import deepcopy
import os
import sys
import multiprocessing as mp
import argparse
from copy import deepcopy
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from esox.fast_io import read_fast, read_fast5
from esox.resquiggle import resquiggle_read_normalized, seq_to_signal
from esox.normalization import med_mad, rescale_lstsq

def listener_writer(queue, output_file):

    while True:
        with open(output_file, 'a') as out_f:
            m = queue.get()
            if m == 'kill':
                break
            else:
                out_f.write(m)


def resquiggle_oligo(read_id, fast5_file, reference_seq, queue):

    read_data = read_fast5(fast5_file, read_ids = read_id)[read_id]
    orig_reference_seq = deepcopy(reference_seq)
    reference_seq = reference_seq.replace('o', 'G')
    reference_seq = reference_seq.upper()

    raw_signal = read_data.raw
    med, mad = med_mad(raw_signal, factor = 1)
    norm_signal = (raw_signal - med)/mad

    try:
        resquiggle_results = resquiggle_read_normalized(
            read_id = read_id, 
            raw_signal = raw_signal, 
            genome_seq = reference_seq, 
            norm_signal = norm_signal,
        )
    except:
        return None

    seg_table = resquiggle_results.segs
    expected_signal = seq_to_signal(reference_seq)
    segment_lengths = seg_table[1:] - seg_table[:-1]

    oxog_pos = np.where(np.array(list(orig_reference_seq[2:-3])) == 'o')[0]
    non_mod_signal_coords = list()
    st = 0
    for op in oxog_pos:
        non_mod_signal_coords.append((st, op-2))
        st = op + 2
    non_mod_signal_coords.append((st, len(segment_lengths)))

    seg_table = resquiggle_results.segs
    expected_signal = seq_to_signal(reference_seq)
    segment_lengths = seg_table[1:] - seg_table[:-1]

    assert len(segment_lengths) == len(expected_signal)

    expected_long_signal = np.zeros((np.sum(segment_lengths),), dtype = float)
    c = 0
    for sl, es in zip(segment_lengths, expected_signal):

        expected_long_signal[c:sl+c] = es
        c += sl


    rescale_st = resquiggle_results.read_start_rel_to_raw
    rescale_nd = rescale_st + len(expected_long_signal)
    raw_rescale = read_data.raw[rescale_st:rescale_nd]

    expected_long_signal_non_mod = list()
    raw_rescale_non_mod = list()
    for s, n in non_mod_signal_coords:
        expected_long_signal_non_mod.append(
            expected_long_signal[seg_table[s]:seg_table[n]]
        )
        raw_rescale_non_mod.append(
            raw_rescale[seg_table[s]:seg_table[n]]
        )
    expected_long_signal_non_mod = np.concatenate(expected_long_signal_non_mod)
    raw_rescale_non_mod = np.concatenate(raw_rescale_non_mod)

    new_med, new_mad = rescale_lstsq(raw_rescale_non_mod, expected_long_signal_non_mod, med, mad)

    write_str = '>'+read_id+';'+str(resquiggle_results.read_start_rel_to_raw)+';'+str(new_med)+';'+str(new_mad)+'\n'+",".join(seg_table.astype(str).tolist())+'\n'
    queue.put(write_str)

    return None
    

def main(ref_file, fast5_file, output_file, n_cores):


    manager = mp.Manager() 
    queue = manager.Queue()  # write queue
    pool = mp.Pool(n_cores + 1) # pool for multiprocessing
    p = mp.Process(target = listener_writer, args = (queue, output_file))
    p.start()

    references = read_fast(ref_file)

    jobs = list()
    for read_id, reference_seq in references.items():

        read_id = read_id.split(';')[0]
        
        job = pool.apply_async(resquiggle_oligo, (read_id, fast5_file, reference_seq, queue))
        jobs.append(job)

    for job in tqdm(jobs):
        job.get()

    queue.put('kill')

    while not queue.empty():
        time.sleep(1)
    
    pool.close()
    pool.join()
    p.join()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Analyze the oligo composition of the reads based on basecalls')
    parser.add_argument('--ref-file', type=str,
                        help='Fasta file with oligo concatemers references')
    parser.add_argument('--fast5-file', type=str,
                        help='Fast5 file with raw data')
    parser.add_argument('--output-file', type=str,
                        help='Output txt file')
    parser.add_argument('--n-cores', type=int, default=1,
                        help='Number of parallel processes')
    
    args = parser.parse_args()

    if not os.path.isfile(args.ref_file):
        raise FileNotFoundError("Reference file not found")
    
    if not os.path.isfile(args.fast5_file):
        raise FileNotFoundError("Fast5 file not found")
    
    if not Path(args.output_file).parent.is_dir():
        print("The directory of the output path does not exist")

    if os.path.exists(args.output_file):
        raise FileExistsError("Output file already exists")
    

    main(
        ref_file = args.ref_file, 
        fast5_file = args.fast5_file,
        output_file = args.output_file, 
        n_cores = args.n_cores,
    )