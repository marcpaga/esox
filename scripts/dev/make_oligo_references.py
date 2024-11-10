import os
import sys
import time
import multiprocessing as mp
import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from esox.fast_io import read_fast
from esox.constants import (
    HEAD_ADAPTER,
    TAIL_ADAPTER,
    POS_PHRED,
)

def listener_writer(queue, output_file):

    while True:
        with open(output_file, 'a') as out_f:
            m = queue.get()
            if m == 'kill':
                break
            else:
                out_f.write(m)

def reconstruct_read_blocks(read_df):

    constr = list()

    for i, (_, row) in enumerate(read_df.iterrows()):
        if i == 0:
            prev_nd = row.basecalls_nd
            constr.append(row.ref)
            continue

        if abs(row.basecalls_st - prev_nd) > 15:
            constr.append('empty')

        constr.append(row.ref)
        prev_nd = row.basecalls_nd

    for i, c in enumerate(constr):
        if c == 'empty':
            if i == 1:
                if constr[i-1].endswith('_adapter'):
                    constr[i] = constr[i+1]
            if constr[i-1] == constr[i+1]:
                constr[i] = constr[i-1]+'_rand'
                continue

            if i == len(constr)-2:
                if constr[i+1].endswith('_adapter'):
                    constr[i] = constr[i-1]

    return constr

def make_read_reference(read_df, oligo_references, fasta_queue):

    refs_in_read = list()
    for r in np.unique(read_df['ref']):
        if r.startswith('oxog_'):
            refs_in_read.append(r)

    if len(refs_in_read) > 2:
        return None

    constr = reconstruct_read_blocks(read_df)

    # solve chimeras
    refs_in_read, refs_counts = np.unique(read_df['ref'], return_counts = True)
    longest_ref = refs_in_read[np.argmax(refs_counts)]

    read_df = read_df[read_df['ref'] == longest_ref]

    constr = reconstruct_read_blocks(read_df)

    if 'empty' in constr:
        return None

    
    row_counter = 0
    final_ref = ""
    phredq_pos_ref = ""
    ref_seq = list()
    for i in range(len(constr)):
        c = constr[i]
        
        if c.endswith('_rand'):
            ref_block = oligo_references[c.replace('_rand', '')]
            n_bases = 'NNNNN'
        else:
            ref = read_df.iloc[row_counter]['ref']
            n_bases = read_df.iloc[row_counter]['n_bases']
            ref_block = oligo_references[ref]
            row_counter += 1

        if c.startswith('oxog_'):
            phredq_pos_ref += POS_PHRED[:len(ref_block)]
        else:
            phredq_pos_ref += POS_PHRED[-len(ref_block):]

        ref_seq.append(c)

        n_counter = 0
        for r in ref_block:
            if r == 'N':
                if n_bases[n_counter] == 'N':
                    r = random.choice(['a', 'c', 'g', 't'])
                else:
                    r = n_bases[n_counter]
                n_counter += 1

            final_ref += r
    
    write_str = '>'+str(np.unique(read_df['read_id'])[0]) + '\n' + final_ref + '\n'
    fasta_queue.put(write_str)

def main(mapped_file, oligo_ref_file, output_file, n_cores):


    manager = mp.Manager() 
    fasta_queue = manager.Queue()  # write queue
    pool = mp.Pool(n_cores + 2) # pool for multiprocessing
    p = mp.Process(target = listener_writer, args = (fasta_queue, output_file))
    p.start()

    mapped_df = pd.read_csv(mapped_file, header = 0, index_col = None)
    mapped_df = mapped_df.sort_values(['read_id', 'basecalls_st'])
    mapped_df = mapped_df.reset_index()

    oligo_references = read_fast(oligo_ref_file)
    oligo_references['head_adapter'] = HEAD_ADAPTER
    oligo_references['tail_adapter'] = TAIL_ADAPTER

    mapped_oligo_df = mapped_df[(mapped_df.ref.str.startswith('oxog_')) & (mapped_df['score'] >= 30)]
    mapped_adapter_df = mapped_df[mapped_df.ref.str.endswith('_adapter')]
    qual_mapped_df = pd.concat([mapped_oligo_df, mapped_adapter_df])
    qual_mapped_df = qual_mapped_df.sort_values(['read_id', 'basecalls_st'])
    qual_mapped_df = qual_mapped_df.reset_index()


    read_ids = np.unique(qual_mapped_df['read_id'])

    jobs = list()
    for read_id in read_ids:
        
        st = qual_mapped_df.read_id.searchsorted(read_id, side = 'left')
        nd = qual_mapped_df.read_id.searchsorted(read_id, side = 'right')
        read_df = qual_mapped_df[st:nd]

        if len(read_df) < 5:
            continue
        
        job = pool.apply_async(make_read_reference, (read_df, oligo_references, fasta_queue))
        jobs.append(job)

    for job in tqdm(jobs):
        job.get()


    fasta_queue.put('kill')
    while not fasta_queue.empty():
        time.sleep(1)

    pool.close()
    pool.join()
    p.join()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Analyze the oligo composition of the reads based on basecalls')
    parser.add_argument('--mapped-file', type=str,
                        help='File with the oligo to basecalls mapping')
    parser.add_argument('--ref-file', type=str,
                        help='Fasta file with oligo references')
    parser.add_argument('--output-file', type=str,
                        help='Dir where to save the reports')
    parser.add_argument('--n-cores', type=int, default=1,
                        help='Number of parallel processes')
    
    args = parser.parse_args()
    
    main(
        mapped_file = args.mapped_file, 
        oligo_ref_file = args.ref_file,
        output_file = args.output_file, 
        n_cores = args.n_cores,
    )

