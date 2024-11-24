import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from esox.models.remora import RemoraModel

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data-path', type=str, required=True)
    argparser.add_argument('--output-path', type=str, required=True)
    argparser.add_argument('--batch_size', type=int, default=256)
    argparser.add_argument('--check_every', type=int, default=100)
    args = argparser.parse_args()

    #throw error if output path does not exist
    if not os.path.exists(args.output_path):
        raise ValueError('Output path does not exist')
    
    # throw error if no gpu
    if not torch.cuda.is_available():
        raise ValueError('No GPU available')
    
    DEVICE = torch.device('cuda:0')

    remora_model = RemoraModel(
        device = DEVICE,
        dataset = None,
        size = 256, 
        dropout = 0, 
        use_raw = True,
        use_seq_1 = True,
        use_expected_signal_1 = True,
    )
    remora_model.to(DEVICE)

    crossentropy_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 0.01]).to(DEVICE))
    optimizer = torch.optim.Adam(remora_model.parameters(), lr=0.001)
    num_training_steps = 0

    for npz_file in os.listdir(args.data_path):

        if not npz_file.endswith('.npz'):
            continue
        
        print(f'Using {npz_file}')
        data = np.load(os.path.join(args.data_path, npz_file))
        loaded_data = {}
        for k in data.keys():
            loaded_data[k] = data[k]
        

        num_samples = loaded_data['x'].shape[0]
        sample_idxs = np.arange(num_samples)
        np.random.shuffle(sample_idxs)

        for i in range(0, num_samples, args.batch_size):
            batch_idxs = sample_idxs[i:i+args.batch_size]
            batch_data = {k: torch.tensor(loaded_data[k][batch_idxs]) for k in loaded_data.keys()}
            batch_data['s1'][batch_data['s1'] == 5] = 3
        
            logits = remora_model.train_step(batch_data)
            loss = crossentropy_loss(logits, batch_data['y'].to(int).to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_training_steps += 1
        
            if num_training_steps % args.check_every == 0:
                print(f'Loss: {loss.item()}')
                print(f'Accuracy: {torch.mean((torch.argmax(logits, dim=1) == batch_data["y"].to(int).to(DEVICE)).float()).item()}')
                print(f'Num training steps: {num_training_steps}')
                print('')


    print('Saving model')
    torch.save(remora_model.state_dict(), os.path.join(args.output_path, 'remora_model.pth'))