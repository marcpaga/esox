import os
import sys
import argparse
import json

import torch
import pandas as pd
import numpy as np
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from esox.models.remora import RemoraModel
from esox.dataclasses import RemoraDataset
from esox.constants import DECODING_DICT_CRF
from esox.fast_io import read_fast

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help='Path with npz AND fastq files')
    parser.add_argument("--output-path", type=str, help='txt file, or dir, to save mod called files')
    parser.add_argument("--model-file", type=str, help='Checkpoint file')
    parser.add_argument("--batch-size", type=int, default = 1024, help='batch size')
    parser.add_argument("--device", type=str, default='cuda:0', help='cpu or cuda device')
    parser.add_argument("--progress-bar", action='store_true', help='If given, print a progress bar for each file')
    args = parser.parse_args()

    npz_files = list()
    for f in os.listdir(args.input_path):
        if f.endswith('.npz'):
            npz_files.append(os.path.join(args.input_path, f))
    print('Found a total of {} npz files'.format(len(npz_files)))

    fastq_files = list()
    for npz_file in npz_files:
        fastq_file = os.path.join(args.input_path, os.path.basename(npz_file).replace('.npz', '.fastq'))

        assert os.path.isfile(fastq_file), "Fastq file {} not found".format(fastq_file)
        fastq_files.append(fastq_file)

    print('Found all relevant fastq files')

    print('Loading Remora model')
    with open('/hpc/compgen/users/mpages/remox2/remox/models/remora_config.json', 'r') as model_configs:
        model_configs = json.load(model_configs)

    remora_model = RemoraModel(
        device = args.device,
        dataset = None,
        size = 256, 
        dropout = 0, 
        **model_configs['single_rse'],
    )
    checkpoint = torch.load(args.model_file)

    if 'model_state' in checkpoint.keys():
        remora_model.load_state_dict(checkpoint['model_state'])
    else:
        remora_model.load_state_dict(checkpoint)    
    remora_model = remora_model.to(args.device)

    remora_dataset = RemoraDataset(
        data_dir = '.', 
        non_masked_bases = 3, 
        seed = 0,
        primary_name = 'primary', 
        secondary_name = None,
        inference_mode = True,
    )

    for npz_file, fastq_file in zip(npz_files, fastq_files):

        mod_output_file = os.path.join(args.output_path, os.path.basename(npz_file).replace('.npz', '.txt'))
        if os.path.isfile(mod_output_file):
            print('Skipping file {}'.format(npz_file))
            continue

        print('Processing file: {}'.format(npz_file))
        basecalls_dict = read_fast(fastq_file)
        arr = np.load(npz_file)

        read_ids = arr['read_id']
        basecalls_pos = arr['basecall_pos']

        data = dict()
        for k in arr.keys():
            if k in ['read_id', 'basecall_pos']:
                continue
            data[k] = arr[k]

        results = np.zeros((arr['x'].shape[0], ), dtype=float)

        total_samples = arr['x'].shape[0]
        batch_size = args.batch_size

        ss = torch.arange(0, total_samples, batch_size)
        nn = ss + batch_size

        df_list = list()
        for s, n in tqdm(zip(ss, nn), total=len(ss), leave=False, disable=not args.progress_bar):

            batch = list()
            for i in range(s, n):
                if i == total_samples:
                    break
                batch.append(remora_dataset.get_data(data, i))
            batch = default_collate(batch)
            
            k5 = ["".join(l) for l in np.vectorize(DECODING_DICT_CRF.get)(batch['s1'].cpu()).tolist()]
            
            oxog_scores = torch.nn.functional.softmax(remora_model.predict_step(batch), -1)[:, 1].cpu().numpy()
            results[s:n] = oxog_scores

        results= trunc(results, decs=4)

        mer5 = list()
        for read_id, bp in zip(read_ids, basecalls_pos):

            seq = basecalls_dict[read_id][0]
            st = bp - 2
            if bp < 0:
                bp = 0
            nd = bp + 3
            if bp > len(seq):
                bp = len(seq)

            assert seq[bp] == 'G'

            mer5.append(seq[st:nd])

        df = {
            'read_id': read_ids,
            'basecalls_pos': basecalls_pos,
            'oxog_score': results,
            '5mer': np.array(mer5),
        }

        pd.DataFrame(df).to_csv(
            mod_output_file,
            header=True,
            sep='\t',
            index=False,
        )

