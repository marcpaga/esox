import os
import sys
import argparse
from copy import deepcopy

import torch
import numpy as np
from tombo.tombo_helper import TomboError
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from esox.models.bonito import BonitoModel
from esox.fast_io import read_fast, read_fast5
from esox.normalization import rescale_data
from esox.remora_utils import (
    find_cut_points_jit, 
    cut_data_jit, 
    bonito_output_to_resquiggled_mapping,
    bonito_basecall,
)

def chunk(signal, chunksize, overlap):
    """
    Convert a read into overlapping chunks before calling
    The first N datapoints will be cut out so that the window ends perfectly
    with the number of datapoints of the read.
    """
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)

    T = signal.shape[0]
    if chunksize == 0:
        chunks = signal[None, :]
        stub = 0
    elif T < chunksize:
        chunks = torch.nn.functional.pad(signal, (chunksize - T, 0))[None, :]
        stub = 0
    else:
        stub = (T - overlap) % (chunksize - overlap)
        chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)
    
    return chunks, stub

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast5-path", type=str, help='Path to fast5 file or dir with fast5 files')
    parser.add_argument("--fastq-path", type=str, help='Path to fastq file or dir with fastq files')
    parser.add_argument("--output-path", type=str, help='Path to dir where to save the output files')
    parser.add_argument("--model-file", type=str, help='Path to Bonito model file')
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--progress-bar", action='store_true', help='If given, print a progress bar for each file')
    parser.add_argument("--demo", action='store_true', help='Just process 100 first reads, for demo purposes')
    args = parser.parse_args()


    print('GPU available: {}'.format(torch.cuda.get_device_name(0)))
    print('Using device: {}'.format(args.device))

    print('Finding files to basecall')
    fast5_files = list()
    if os.path.isfile(args.fast5_path):
        fast5_files.append(args.fast5_path)
    elif os.path.isdir(args.fast5_path):
        for f in os.listdir(args.fast5_path):
            if f.endswith('.fast5'):
                fast5_files.append(os.path.join(args.fast5_path, f))

    print('Found a total of {} fast5 files'.format(len(fast5_files)))

    fastq_files = list()
    if os.path.isfile(args.fastq_path):
        fastq_files.append(args.fastq_path)
    else:
        for fast5_file in fast5_files:
            fastq_file = os.path.join(args.fastq_path, os.path.basename(fast5_file).replace('.fast5', '.fastq'))

            assert os.path.isfile(fastq_file), "Fastq file {} not found".format(fastq_file)
            fastq_files.append(fastq_file)

    print('Found all relevant fastq files')

    print('Loading basecalling model')
    # load bonito models
    primary_model = BonitoModel(
        size = 384,
        device = args.device,
        dataset = None, 
    )

    primary_model = primary_model.to(args.device)
    checkpoint = torch.load(args.model_file)
    if 'model_state' in checkpoint.keys():
        primary_model.load_state_dict(checkpoint['model_state'])
    else:
        primary_model.load_state_dict(checkpoint)
    primary_model.to(args.device)

    print('Begin basecalling')
    
    window_size = 100
    half_window = window_size // 2

    for fast5_file, fastq_file in zip(fast5_files, fastq_files):

        fastq_output_file = open(os.path.join(args.output_path, os.path.basename(fast5_file).replace('.fast5', '.fastq')), 'w')
        npz_output_file = os.path.join(args.output_path, os.path.basename(fast5_file).replace('.fast5', '.npz'))

        if os.path.isfile(npz_output_file):
            print('Skipping file {}'.format(fast5_file))
            continue

        print('Processing file: {}'.format(fast5_file))

        # read the files
        basecalls_dict = read_fast(fastq_file)
        fast5_dict = read_fast5(fast5_file)

        # where to store the prepared data for mod calling
        array_keeper = {
            'x': list(),
            'e1': list(), # expected
            's1': list(), # predicted sequence
            'p1': list(), # phredq scores
            'basecall_pos': list(),
            'read_id': list(),
        }
        
        # process each read
        for i, read_id in tqdm(enumerate(fast5_dict.keys()), total=len(fast5_dict), leave=False, disable=not args.progress_bar):

            if args.demo:
                if i > 100:
                    break

            read_data = fast5_dict[read_id]
            try:
                basecalls = basecalls_dict[read_id][0]
            except KeyError:
                continue

            # normalize and chunk the fast5 raw data
            try:
                norm_signal = rescale_data(read_data, basecalls, read_id)
            except TomboError as e:
                fastq_output_file.write(">{}\n{}\n+\n{}\n".format(
                    read_id, 
                    "",
                    ""
                ))
                continue

            chunks, stub = chunk(norm_signal, 2000, 400)
            num_chunks = chunks.shape[0]

            original_len = len(norm_signal)-stub

            primary_basecalls, primary_phredq, _ = bonito_basecall(
                x = chunks, 
                model = primary_model, 
                batch_size = args.batch_size, 
                orig_len = original_len,
                whole_read = True,
            )
        
            if len(primary_basecalls) == 0:
                fastq_output_file.write(">{}\n{}\n+\n{}\n".format(
                    read_id, 
                    "",
                    ""
                ))
                continue
    
            sequences_arr = dict()
            phredq_arr = dict()
            expected_arr = dict()
            successful_resquiggle = True
            for name, basecalls, phredq in zip(['1'], [primary_basecalls], [primary_phredq]):
                try:
                    sequences_arr[name], phredq_arr[name], expected_arr[name] = bonito_output_to_resquiggled_mapping(
                        norm_signal[stub:], basecalls, phredq
                    )
                except TomboError as e:
                    fastq_output_file.write(">{}\n{}\n+\n{}\n".format(
                        read_id, 
                        "",
                        ""
                    ))
                    continue
                

            g_calls = np.where(sequences_arr['1'] == 3)[0]

            if len(g_calls) < 2:
                fastq_output_file.write(">{}\n{}\n+\n{}\n".format(
                    read_id, 
                    "",
                    ""
                ))
                continue

            cuts = np.zeros((len(g_calls), 2), dtype=np.int32)
            cuts_with_coords = np.zeros((len(g_calls), ), dtype=bool)

            cuts, _ = find_cut_points_jit(
                g_calls, original_len, half_window, cuts, cuts_with_coords
            )
            if cuts.shape[0] < 2:
                fastq_output_file.write(">{}\n{}\n+\n{}\n".format(
                    read_id, 
                    "",
                    ""
                ))
                continue

            xcut = np.zeros((cuts.shape[0], window_size), dtype = np.float32)
            xcut = cut_data_jit(
                cuts,
                norm_signal[stub:],
                xcut,
            )
            array_keeper['x'].append(xcut)

            for name, path in sequences_arr.items():
                scut = np.zeros((cuts.shape[0], window_size), dtype = np.int8)
                scut = cut_data_jit(cuts, path, scut)
                array_keeper['s'+name].append(scut)

            for name, exp in expected_arr.items():
                ecut = np.zeros((cuts.shape[0], window_size), dtype = np.float32)
                ecut = cut_data_jit(cuts, exp, ecut)
                array_keeper['e'+name].append(ecut)

            for name, phr in phredq_arr.items():
                pcut = np.zeros((cuts.shape[0], window_size), dtype = np.int8)
                pcut = cut_data_jit(cuts, phr, pcut)
                array_keeper['p'+name].append(pcut)

            primary_basecalls = np.array(list(primary_basecalls))
            primary_basecalls_oxo = deepcopy(primary_basecalls)
            primary_basecalls_oxo[:2] = 'N'
            primary_basecalls_oxo[-2:] = 'N'
            basecall_pos = np.where(primary_basecalls_oxo == 'G')[0][cuts_with_coords]
            array_keeper['basecall_pos'].append(basecall_pos)
            array_keeper['read_id'].append(np.array([read_id]*cuts.shape[0]))

            fastq_output_file.write(">{}\n{}\n+\n{}\n".format(
                read_id, 
                "".join(primary_basecalls.tolist()),
                "".join(primary_phredq.tolist())
            ))

        for k, v in array_keeper.items():
            array_keeper[k] = np.concatenate(v, axis=0)

        np.savez_compressed(
            npz_output_file,
            **array_keeper,
        )

        fastq_output_file.close()