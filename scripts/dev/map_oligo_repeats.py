import sys
import os
import time
from copy import deepcopy
import multiprocessing as mp
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from esox.fast_io import read_fast
from esox.constants import (
    FWD_OLIGO_DESIGN, 
    REV_OLIGO_DESIGN, 
    LOCAL_ALIGN_FUNCTION, 
    MATRIX, 
    GAP_OPEN_PENALTY, 
    GAP_EXTEND_PENALTY,
    PHRED_LETTERS,
    HEAD_ADAPTER,
    TAIL_ADAPTER,
)


def make_align_arr(alignment, phredq, design_str, query_st, reference_str):

    arr = np.full((5, len(alignment.traceback.query)), '', dtype=str)
    arr[0] = list(alignment.traceback.ref)
    arr[1] = list(alignment.traceback.comp)
    arr[2] = list(alignment.traceback.query)

    c = 0
    for i in range(arr.shape[1]):
        if arr[2, i] == '-':
            continue
        else:
            try:
                arr[3, i] = phredq[c]
            except IndexError:
                break
            c += 1

    q = np.where(arr[1] != ' ')[0]
    try:
        qst = q[0] + query_st
    except IndexError:
        return None, None
    qnd = q[-1] + query_st

    qnd += np.sum(arr[2, qst:qnd] == '-')

    dots = np.where(arr[1] == '.')[0]
    doubledots = np.where(arr[1] == ':')[0]
    bars = np.where(arr[1] == '|')[0]
    aligns = np.concatenate([
        dots, doubledots, bars
    ])
    
    arr = arr[:, np.min(aligns):np.max(aligns)+1]

    aligned_ref = "".join(arr[0].tolist()).replace('-', '')
    ref_to_ref_align = LOCAL_ALIGN_FUNCTION(
        s1 = reference_str, 
        s2 = aligned_ref, 
        open = GAP_OPEN_PENALTY,
        extend = GAP_EXTEND_PENALTY,
        matrix = MATRIX
    )

    ref_cut = 0
    for b in ref_to_ref_align.traceback.ref:
        if b == '-':
            ref_cut += 1
        else:
            break
    

    c = 0 + ref_cut
    for i in range(arr.shape[1]):
        if arr[0, i] == '-':
            continue
        else:
            arr[4, i] = design_str[c]
            c += 1

    # check that the design_str aligns properly with the reference
    try:
        assert  np.all(arr[4, np.where(arr[0] == 'N')[0]] == 'N')
    except AssertionError:
        return None, None

    return arr, (qst, qnd)

def calculate_score(align_arr):
    return np.sum(align_arr[1, align_arr[4] != 'M'] == '|')
    

def find_repeats(original_basecalls, oligo_references, read_id, phredq_txt, queue):

    basecalls = deepcopy(original_basecalls)
    max_iters = (len(basecalls) // 46) + 2
    current_iter = 0
    num_non_z = len(basecalls)
    list_of_results = list()

    while current_iter < max_iters:

        best_ref = ''
        best_score = 0
        best_arr = None
        best_coords = None

        if num_non_z < 15:
            break

        for ref_k in oligo_references.keys():

            aln = LOCAL_ALIGN_FUNCTION(
                s1 = basecalls, 
                s2 = oligo_references[ref_k].replace('o', 'K'), 
                open = GAP_OPEN_PENALTY,
                extend = GAP_EXTEND_PENALTY,
                matrix = MATRIX
            )
            if ref_k.endswith('_fwd'):
                design_str = FWD_OLIGO_DESIGN[aln.cigar.beg_ref:aln.end_ref+1]
            elif ref_k.endswith('_rev'):
                design_str = REV_OLIGO_DESIGN[aln.cigar.beg_ref:aln.end_ref+1]
            elif ref_k.endswith('_adapter'):
                pass
            else:
                raise ValueError("Invalid reference name: {}".format(ref_k))

            cigar = aln.cigar
            arr, coords = make_align_arr(aln, phredq_txt, design_str, cigar.beg_query, oligo_references[ref_k].replace('o', 'K'))

            if arr is None:
                continue

            score = calculate_score(arr)

            if score > best_score:
                best_score = score
                best_ref = ref_k
                best_arr = arr
                best_coords = coords

        if best_arr is None:
            break

        n_bases = best_arr[2, best_arr[0] == 'N']
        unkown_bases = np.sum(n_bases == '-')
        
        result = {
            'read_id': read_id,
            'ref': best_ref,
            'score': best_score,
            'basecalls_st': best_coords[0],
            'basecalls_nd': best_coords[1],
            'n_bases': "".join(n_bases.tolist()).replace('-', 'N'),
            'unkown_n_bases': unkown_bases,
            'len_basecalls': len(basecalls),
        }
        for i, q in enumerate(best_arr[3, best_arr[0] == 'N']):
            try:
                q = PHRED_LETTERS[q]
            except KeyError:
                q = 0
            result['n_q'+str(i)] = q

        list_of_results.append(result)


        basecalls = basecalls[:best_coords[0]] + 'Z'* (best_coords[1]-best_coords[0]) + basecalls[best_coords[1]:]
        num_non_z = np.sum(np.array(list(basecalls)) != 'Z')


        current_iter += 1

    queue.put(pd.DataFrame(list_of_results))

def listener_writer(queue, output_file):

    while True:
        m = queue.get()
        if isinstance(m, str) and m == 'kill':
            break
        m.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index = False)
        


def main(fastq_dir, ref_file, output_file, n_cores, overwrite):

    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            df = pd.read_csv(output_file, header = 0, index_col = None)
            processed_read_ids = set(np.unique(df['read_id']))
    else:
        processed_read_ids = set()
        

    oligo_references = read_fast(ref_file)
    oligo_references['head_adapter'] = HEAD_ADAPTER
    oligo_references['tail_adapter'] = TAIL_ADAPTER

    manager = mp.Manager() 
    queue = manager.Queue()  # write queue
    pool = mp.Pool(n_cores + 2) # pool for multiprocessing
    p = mp.Process(target = listener_writer, args = (queue, output_file))
    p.start()

    jobs = list()
    for fastq_file in os.listdir(fastq_dir):
        if not fastq_file.endswith('.fastq'):
            continue
        fastq = read_fast(os.path.join(fastq_dir, fastq_file))
    
        for _, (k, v) in enumerate(fastq.items()):
            read_id = k
            if read_id in processed_read_ids:
                continue
            original_basecalls = v[0]
            if len(original_basecalls) < 60: # skip very short reads
                continue
            if len(original_basecalls) > 20000:
                continue
            phredq_txt = v[2]
            job = pool.apply_async(find_repeats, (original_basecalls, oligo_references, read_id, phredq_txt, queue))
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
    parser.add_argument('--fastq-dir', type=str,
                        help='Dir where the fastq files are')
    parser.add_argument('--ref-file', type=str,
                        help='Fasta file with oligo references')
    parser.add_argument('--output-file', type=str,
                        help='Dir where to save the reports')
    parser.add_argument('--n-cores', type=int, default=1,
                        help='Number of parallel processes')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite files if output file exists')
    
    args = parser.parse_args()
    
    main(
        fastq_dir = args.fastq_dir, 
        ref_file = args.ref_file, 
        output_file = args.output_file, 
        n_cores = args.n_cores, 
        overwrite = args.overwrite
    )

