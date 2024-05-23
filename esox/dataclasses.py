import os
import random
from copy import deepcopy
import logging

import numpy as np
from torch.utils.data import Dataset, Sampler
from numba import jit
from tqdm import tqdm

from esox.constants import DECODING_DICT_CRF, ENCODING_DICT_CRF, BLANK_TOKEN
#from esox.utils import read_metadata

class BaseDataset(Dataset):

    """Base dataset class that contains Nanopore data
    
    The simplest class that handles a hdf5 file that has two datasets
    named 'x' and 'y'. The first one contains an array of floats with
    the raw data normalized. The second one contains an array of 
    byte-strings with the bases appended with ''.
    
    This dataset already takes case of shuffling, for the dataloader set
    shuffling to False.
    
    Args:
        data (str): dir with the npz files
        decoding_dict (dict): dictionary that maps integers to bases
        encoding_dict (dict): dictionary that maps bases to integers
        split (float): fraction of samples for training
        randomizer (bool): whether to randomize the samples order
        seed (int): seed for reproducible randomization
        s2s (bool): whether to encode for s2s models
        token_sos (int): value used for encoding start of sequence
        token_eos (int): value used for encoding end of sequence
        token_pad (int): value used for padding all the sequences (s2s and not s2s)
    """

    def __init__(
        self, 
        data_dir, 
        split = 0.95, 
        shuffle = True, 
        seed = None,
    ):
        super(BaseDataset, self).__init__()
        

        self.data_dir = data_dir
        self.split = split
        self.shuffle = shuffle
        self.seed = seed

        if os.path.isdir(self.data_dir):
            self.files_list = self._find_files()
        elif os.path.isfile(self.data_dir):
            self.files_list = [self.data_dir]
        else:
            raise ValueError('data_dir must be a path to a file or a dir, given: {}'.format(self.data_dir))

        self.num_samples_per_file = self._get_samples_per_file()
        self.train_files_idxs = set()
        self.validation_files_idxs = set()
        self.train_idxs = list()
        self.validation_idxs = list()
        self.samplers = dict()
        self._split_train_validation()
        self._get_samplers()
        
        self.loaded_train_data = None
        self.loaded_validation_data = None
        self.current_loaded_train_idx = None
        self.current_loaded_validation_idx = None

    def __len__(self):
        """Number of samples
        """
        return len(self.validation_idxs) + len(self.train_idxs)
        
    def __getitem__(self, idx):
        """Get a set of samples by idx
        
        If the datafile is not loaded it loads it, otherwise
        it uses the already in memory data.
        
        Returns a dictionary
        """
        if idx[0] in self.train_files_idxs:
            if idx[0] != self.current_loaded_train_idx:
                self.loaded_train_data = self.load_file_into_memory(idx[0])
                self.current_loaded_train_idx = idx[0]
            return self.get_data(data_dict = self.loaded_train_data, idx = idx[1])
        elif idx[0] in self.validation_files_idxs:
            if idx[0] != self.current_loaded_validation_idx:
                self.loaded_validation_data = self.load_file_into_memory(idx[0])
                self.current_loaded_validation_idx = idx[0]
            return self.get_data(data_dict = self.loaded_validation_data, idx = idx[1])
        else:
            raise IndexError('Given index not in train or validation files indices: ' + str(idx[0]))
    
    
    def _find_files(self):
        """Finds list of files to read
        """
        l = list()
        for f in os.listdir(self.data_dir):
            if f.endswith('.npz'):
                l.append(f)
        l = sorted(l)
        return l
    
    def _get_samples_per_file(self):
        """Gets the number of samples per file from the file name
        """
        print('Getting dataset sizes')
        l = list()
        failed_files = False
        for f in tqdm(self.files_list):
            try:
                metadata = read_metadata(os.path.join(self.data_dir, f))
            except:
                print('Error reading: {}'.format(os.path.join(self.data_dir, f)))
                failed_files = True
                continue

            l.append(metadata[0][1][0]) # [array_num, shape, first elem shape]

        if failed_files:
            raise ValueError("Some files are corrupted")
        return l
    
    def _split_train_validation(self):
        """Splits datafiles and idx for train and validation according to split
        """
        
        # split train and validation data based on files
        num_train_files = int(len(self.files_list) * self.split)
        
        files_idxs = list(range(len(self.files_list)))
        if self.shuffle:
            if self.seed:
                random.seed(self.seed)
            random.shuffle(files_idxs)
            
        self.train_files_idxs = set(files_idxs[:num_train_files])
        self.validation_files_idxs = set(files_idxs[num_train_files:])
        
        # shuffle indices within each file and make a list of indices (file_idx, sample_idx)
        # as tuples that can be iterated by the sampler
        for idx in self.train_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.train_idxs.append((idx, i))
        
        for idx in self.validation_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.validation_idxs.append((idx, i))
                
        return None
    
    def _get_samplers(self):
        """Add samplers
        """
        for cvset, idxs in zip(
            ['train', 'validation'], 
            [self.train_idxs, self.validation_idxs]
        ):
            self.samplers[cvset] = IdxSampler(idxs, data_source = self)

        return self.samplers
            
    def load_file_into_memory(self, idx):
        """Loads a file into memory and processes it
        """
        raise NotImplementedError
    
    def get_data(self, data_dict, idx):
        """Slices the data for given indices
        """
        raise NotImplementedError
    
    def resample(self):
        self._split_train_validation()
        self._get_samplers()
    

class BaseNanoporeDataset(BaseDataset):
    """Base dataset class that contains Nanopore data
    
    The simplest class that handles a hdf5 file that has two datasets
    named 'x' and 'y'. The first one contains an array of floats with
    the raw data normalized. The second one contains an array of 
    byte-strings with the bases appended with ''.
    
    This dataset already takes case of shuffling, for the dataloader set
    shuffling to False.
    
    Args:
        data (str): dir with the npz files
        decoding_dict (dict): dictionary that maps integers to bases
        encoding_dict (dict): dictionary that maps bases to integers
        split (float): fraction of samples for training
        randomizer (bool): whether to randomize the samples order
        seed (int): seed for reproducible randomization
        s2s (bool): whether to encode for s2s models
        token_sos (int): value used for encoding start of sequence
        token_eos (int): value used for encoding end of sequence
        token_pad (int): value used for padding all the sequences (s2s and not s2s)
    """

    def __init__(
        self, 
        decoding_dict = DECODING_DICT_CRF, 
        encoding_dict = ENCODING_DICT_CRF, 
        token_pad = BLANK_TOKEN,
        randomize_oligo = False,
        randomize_dist = 3,
        *args,
        **kwargs,
    ):
        super(BaseNanoporeDataset, self).__init__(*args, **kwargs)
        
        self.decoding_dict = decoding_dict
        self.encoding_dict = encoding_dict
        self.token_pad = token_pad

        self.randomize_oligo = randomize_oligo
        self.randomize_dist = randomize_dist
        
            
    def load_file_into_memory(self, idx):
        """Loads a file into memory and processes it
        """
        logging.debug('Loading numpy file into memory: {}'.format(os.path.join(self.data_dir, self.files_list[idx])))

        arr = np.load(os.path.join(self.data_dir, self.files_list[idx]))
        x = arr['x']
        y = arr['y']

        return self.process({'x':x, 'y':y})
    
    def get_data(self, data_dict, idx):
        """Slices the data for given indices
        """
        logging.debug('Getting sample: {}'.format(idx))

        if self.randomize_oligo:
            return {'x': data_dict['x'][idx], 'y': data_dict['y'][idx], 'y_rand': data_dict['y_rand'][idx]}
        else:
            return {'x': data_dict['x'][idx], 'y': data_dict['y'][idx]}
    
    def process(self, data_dict):
        """Processes the data into a ready for training format
        """
        
        y = data_dict['y']
        if y.dtype != 'U1':
            y = y.astype('U1')
        
        data_dict['y'], data_dict['y_rand'] = self.encode(y)
        return data_dict
    
    def encode(self, y_arr):
        """Encode the labels
        """
        # preallocate encoded array
        new_y = np.full(y_arr.shape, '', dtype=y_arr.dtype)
        new_y_rand = np.full(y_arr.shape, '', dtype=y_arr.dtype)
        # move all labels to the left
        for i in range(y_arr.shape[0]):

            if self.randomize_oligo:
                y_rand = self.randomize_around_mod_base(y_arr[i])

            bpos = np.where(y_arr[i] != '')[0]
            new_y[i, :len(bpos)] = y_arr[i, bpos]
            if self.randomize_oligo:
                new_y_rand[i, :len(bpos)] = y_rand[bpos]
        new_y = np.vectorize(self.encoding_dict.get)(new_y)
        if self.randomize_oligo:
            new_y_rand = np.vectorize(self.encoding_dict.get)(new_y_rand)
        return new_y, new_y_rand
    
    def encoded_array_to_list_strings(self, y):
        """Convert an encoded array back to a list of strings

        Args:
            y (array): with shape [batch, len]
        """

        y = y.astype(str)
        
        y[y == str(self.token_pad)] = ''
        # replace predictions with bases
        for k, v in self.decoding_dict.items():
            y[y == str(k)] = v

        # join everything
        decoded_sequences = ["".join(i) for i in y.tolist()]
        return decoded_sequences

    def randomize_around_mod_base(self, y):
        """Randomize bases of samples that contain a modified base
        """

        o_num = np.where(y == 'o')[0]
        if len(o_num) == 0:
            return y
        b_num = np.where(y != '')[0]

        oo_num = np.where(np.isin(b_num, o_num))[0]

        d = np.tile(np.arange(len(b_num)), len(oo_num)).reshape(len(oo_num), len(b_num)) - np.repeat(oo_num, len(b_num)).reshape(len(oo_num), len(b_num))
        d = np.abs(d).min(0)
        pos_to_randomize = b_num[d > self.randomize_dist]

        rand_y = deepcopy(y)

        rand_bases = np.random.choice(['A', 'C', 'G', 'T'], size = len(pos_to_randomize))
        rand_y[pos_to_randomize] = rand_bases

        return rand_y




class IdxSampler(Sampler):
    """Sampler class to not sample from all the samples
    from a dataset.
    """
    def __init__(self, idxs, *args, **kwargs):
        super(IdxSampler, self).__init__(*args, **kwargs)
        self.idxs = idxs

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)
    

class BalanceSampler(Sampler):
    """Sampler that balances positives and negatives in an unbalanced dataset.
    Assumes that the positive class is underrepresented

    Args:
        neg_samples_per_file: np.array with the number of negative samples per file
        pos_samples_per_file: np.array with the number of positive samples per file
        balance_mode: 'upsample' for upsampling the positive class to the number
            of negative samples. 'downsample' for downsampling the negative class
            to the number of positive samples.
        reduce_dataset: use only a fraction of the samples
    """
    def __init__(
            self, 
            files_idxs,
            neg_samples_per_file, 
            pos_samples_per_file,
            balance_mode, 
            reduce_dataset = 1.0,
            *args, 
            **kwargs
            ):
        
        super(BalanceSampler, self).__init__(*args, **kwargs)
        self.files_idxs = files_idxs
        self.neg_samples_per_file = neg_samples_per_file
        self.pos_samples_per_file = pos_samples_per_file
        self.balance_mode = balance_mode
        self.reduce_dataset = reduce_dataset

        assert len(neg_samples_per_file) == len(pos_samples_per_file)
        assert np.all(neg_samples_per_file - pos_samples_per_file > 0)

        self.total_files = len(self.files_idxs)
        self.total_neg_samples = neg_samples_per_file.sum()
        self.total_pos_samples = pos_samples_per_file.sum()

        if self.balance_mode == 'upsample':
            self.total_samples = int(self.total_neg_samples * 2 * self.reduce_dataset)
        elif self.balance_mode == 'downsample':
            raise NotImplementedError
            self.total_samples = self.total_pos_samples * 2
        else:
            raise ValueError('Unrecognized balance_mode: {}. Options: "upsample" or "downsample"'.format(balance_mode))

        self.iterator_tracker = np.zeros((len(self.neg_samples_per_file), ), dtype=int)

        self.current_file = 0
        self.current_class = 0
        self.ni = 0
        self.pi = 0
        self.i = 0
        self.neg_idxs, self.pos_idxs = self.prepare_idxs()

    def prepare_idxs(self):
        self.ni = 0
        self.pi = 0
        neg_idxs = np.arange(self.neg_samples_per_file[self.current_file])
        neg_idxs = neg_idxs.astype(int)
        np.random.seed(self.current_file)
        np.random.shuffle(neg_idxs)

        pos_idxs = np.arange(self.pos_samples_per_file[self.current_file]) + neg_idxs.max()
        np.random.seed(self.i)
        np.random.shuffle(pos_idxs)
        pos_idxs = pos_idxs.astype(int)

        neg_idxs = neg_idxs[:int(len(neg_idxs) * self.reduce_dataset)]

        self.neg_idxs = neg_idxs
        self.pos_idxs = pos_idxs
        return neg_idxs, pos_idxs

    def __iter__(self):
        return self

    def __len__(self):
        return self.total_samples
    
    def __next__(self):

        if self.i >= self.total_samples:
            raise StopIteration
        
        if self.balance_mode == 'upsample':
            return self.next_upsample()
        elif self.balance_mode == 'downsample':
            raise NotImplementedError
    
    def next_upsample(self):
        if self.current_class:
            try:
                result = (self.files_idxs[self.current_file], self.pos_idxs[self.pi])
                self.pi += 1
                
            except IndexError:
                # end of positive samples
                # keep track of next negative sample for this file
                self.iterator_tracker[self.current_file] += self.ni
                # not the best solution, but just repeat the first sample
                result = (self.files_idxs[self.current_file], np.random.choice(self.pos_idxs))
                
                self.current_file += 1
                if self.current_file >= self.total_files:
                    self.current_file = 0
                # reset idxs
                self.prepare_idxs()

            self.i += 1
            self.current_class = 0
            
            return result
        
        if not self.current_class:
            try:
                result = (self.files_idxs[self.current_file], self.neg_idxs[self.ni + self.iterator_tracker[self.current_file]])
                self.ni += 1
                
            except IndexError:
                # end of negative samples
                # keep track of next negative sample for this file
                self.iterator_tracker[self.current_file] = self.ni
                # not the best solution, but just repeat the first sample
                result = (self.files_idxs[self.current_file], np.random.choice(self.neg_idxs))
                
                self.current_file += 1
                if self.current_file >= self.total_files:
                    self.current_file = 0
                # reset idxs
                self.prepare_idxs()

            
            self.i += 1
            self.current_class = 1
            
            return result
        
class RemoraDataset(Dataset):

    """
    Args:
        primary_name: name of the primary bonito model
        non_masked_bases: number of bases, on either side, to not be masked
        secondary_name: no of the secondary bonito model
        cd_filter: if given, samples with difference in cd lower than value
            will be excluded
        pos_min_dist: if given, samples with guanine closer than value to an
            8oxodG will be excluded
        balance: if True, then samples are filtered to have the same amount
            of positives and negatives
    """

    def __init__(
        self, 
        data_dir,
        primary_name,
        shuffle = True,
        seed = None,
        non_masked_bases = 2,
        masking_value = 0,
        secondary_name = None,
        cd_filter = np.NINF,
        pos_min_dist = 0,
        balance_mode = 'upsample',
        reduce_dataset = 1.0,
        inference_mode = False,
        additional_keys = list(),
        *args, 
        **kwargs,
    ):

        self.data_dir = data_dir
        self.shuffle = shuffle
        self.seed = seed
        self.non_masked_bases = non_masked_bases
        self.masking_value = masking_value
        self.primary_name = primary_name
        self.secondary_name = secondary_name
        self.cd_filter = cd_filter 
        self.pos_min_dist = pos_min_dist
        self.balance_mode = balance_mode
        self.reduce_dataset = reduce_dataset
        self.inference_mode = inference_mode
        self.additional_keys = additional_keys

        self.expected_batch_keys = ['x', 'y'] + self.additional_keys
        self.key_mapper = {'x':'x', 'y':'y'}
        for k in self.additional_keys:
            self.key_mapper[k] = k
        for k in ['s', 'p', 'e']:
            self.expected_batch_keys.append("{}_{}".format(self.primary_name, k))
            self.key_mapper["{}_{}".format(self.primary_name, k)] = '{}1'.format(k)
            if self.secondary_name is not None:
                self.expected_batch_keys.append("{}_{}".format(self.secondary_name, k))
                self.key_mapper["{}_{}".format(self.secondary_name, k)] = '{}2'.format(k)

        if self.inference_mode:
            for k in ['s', 'p', 'e']:
                self.expected_batch_keys.append("{}{}".format(k, self.primary_name))
                self.key_mapper["{}{}".format(k,self.primary_name)] = '{}1'.format(k)
                if self.secondary_name is not None:
                    self.expected_batch_keys.append("{}{}".format(k,self.secondary_name, ))
                    self.key_mapper["{}{}".format(k,self.secondary_name)] = '{}2'.format(k)

        self.files_list = list()
        for f in os.listdir(self.data_dir):
            if f.endswith('.npz'):
                self.files_list.append(os.path.join(self.data_dir, f))
        self.files_list = sorted(self.files_list)
        
        self.current_loaded_train_idx = -1
        self.current_loaded_validation_idx = -1
        self.current_loaded_test_idx = -1

        if not self.inference_mode:
            
            print('Splitting negative data')
            self._split_cvset_files()
            print('Loading positive data')
            self.positive_data = self.load_positive_data()
            print('Getting sample sizes per file')
            self._get_samples_per_file()
            print('Preparing samplers')
            self._get_samplers()
        
        

        super(RemoraDataset, self).__init__(*args, **kwargs)
        
    def __getitem__(self, idx):
        """Get a set of samples by idx
        
        If the datafile is not loaded it loads it, otherwise
        it uses the already in memory data.

        Idx should be (file_idx, sample_idx)
        
        Returns a dictionary
        """

        # get a positive label sample
        if idx[0] == self.train_pos_file_idx:
            return self.get_data(data_dict = self.positive_data['train'], idx = idx[1])
        if idx[0] == self.validation_pos_file_idx:
            return self.get_data(data_dict = self.positive_data['validation'], idx = idx[1])
        if idx[0] == self.test_pos_file_idx:
            return self.get_data(data_dict = self.positive_data['test'], idx = idx[1])

        
        
        if idx[0] in self.train_files_idxs:
            if idx[0] != self.current_loaded_train_idx:
                self.loaded_train_data = self.load_file_into_memory(idx[0])
                self.current_loaded_train_idx = idx[0]
            return self.get_data(data_dict = self.loaded_train_data, idx = idx[1])
        elif idx[0] in self.validation_files_idxs:
            if idx[0] != self.current_loaded_validation_idx:
                self.loaded_validation_data = self.load_file_into_memory(idx[0])
                self.current_loaded_validation_idx = idx[0]
            return self.get_data(data_dict = self.loaded_validation_data, idx = idx[1])
        elif idx[0] in self.test_files_idxs:
            if idx[0] != self.current_loaded_test_idx:
                self.loaded_test_data = self.load_file_into_memory(idx[0])
                self.current_loaded_test_idx = idx[0]
            return self.get_data(data_dict = self.loaded_test_data, idx = idx[1])
        else:
            raise IndexError('Given index not in train or validation files indices: ' + str(idx[0]))
    
    
    def _split_cvset_files(self):
        """Splits datafiles and idx for train and validation according to split
        """
        
        # split train and validation data based on files
        # substract one for the positive file
        self.train_files_idxs = list()
        self.validation_files_idxs = list()
        self.test_files_idxs = list()
        for i, f in enumerate(self.files_list):
            
            if '_pos_' in f:
                continue

            if 'train_neg' in f:
                self.train_files_idxs.append(i)
            elif 'validation_neg' in f:
                self.validation_files_idxs.append(i)
            elif 'test_neg' in f:
                self.test_files_idxs.append(i)

        self.train_files_idxs = np.array(self.train_files_idxs)
        self.validation_files_idxs = np.array(self.validation_files_idxs)
        self.test_files_idxs = np.array(self.test_files_idxs)
                
        return None   
    
    def resample(self):
        raise self._get_samplers()

    def __len__(self):
        """Number of samples
        """
        return len(self.samplers['train'])

    def load_file_into_memory(self, idx):
        """Loads a file into memory and processes it
        """
    
        #logging.debug('Loading numpy file into memory: {}'.format(os.path.join(self.data_dir, self.files_list[idx])))

        data = dict()
        arr = np.load(os.path.join(self.data_dir, self.files_list[idx]))
        for k in self.expected_batch_keys:
            try:
                data[self.key_mapper[k]] = arr[k]
            except KeyError:
                if k == 'y' and self.inference_mode:
                    continue
                else:
                    raise KeyError('''
                    Expected key {} in the data, but it is not there. 
                    Keys in the data: {}
                    '''.format(k, list(data.keys())))

        return data
    
    def load_positive_data(self):

        pos_data = dict()
        for i, f in enumerate(self.files_list):
            if f.endswith('train_pos_0.npz'):
                pos_data['train'] = self.load_file_into_memory(idx = i)
                self.train_pos_file_idx = i
            elif f.endswith('validation_pos_0.npz'):
                pos_data['validation'] = self.load_file_into_memory(idx = i)
                self.validation_pos_file_idx = i
            elif f.endswith('test_pos_0.npz'):
                pos_data['test'] = self.load_file_into_memory(idx = i)
                self.test_pos_file_idx = i

        return pos_data

    def get_data(self, data_dict, idx, file_idx = None):
        """Slices the data for given indices
        """

        #logging.debug('Getting sample: {}'.format(idx))
        
        

        # implement a method for each key
        # x: nothing
        # e1-e2 -> ce: just substract one from another
        # d1-d2 -> cd: just substract one from another
        # s1-s2 -> cs: substract and make negatives positive + 5 so that all values are positives and different
        # p1-p2 -> cp: same as cs but add 100
        # [all, including new differentials]: mask the data, based on s1

        slice_data = dict()
        # raw data, as is
        slice_data['x'] = data_dict['x'][idx]
        if not self.inference_mode:
            y = data_dict['y'][idx]
            if y < 0:
                y = 0
            slice_data['y'] = y

        for k in self.additional_keys:
            slice_data[k] = data_dict[k][idx]

        # get the mask based on the primary basecalls
        mask_st, mask_nd = self.get_mask(data_dict['s1'][idx], self.non_masked_bases)
 
        # mask the data for the expected, basecalls and phredq
        only_masked_keys = ['e', 's', 'p']
        for k in only_masked_keys:
            slice_data[f'{k}1'] = self.mask_data(data_dict[f'{k}1'][idx], mask_st, mask_nd, self.masking_value)
            if self.secondary_name is not None:
                slice_data[f'{k}2'] = self.mask_data(data_dict[f'{k}2'][idx], mask_st, mask_nd, self.masking_value)
            
        # calculate the difference between raw and expected and mask it because
        # the raw is not masked
        slice_data['d1'] = self.mask_data(
            slice_data['x'] - slice_data['e1'],
            mask_st, mask_nd, self.masking_value
        )
        if self.secondary_name is not None:
            slice_data['d2'] = self.mask_data(
                slice_data['x'] - slice_data['e2'],
                mask_st, mask_nd, self.masking_value
            )

        # the following do not need masking since we work with data that has 
        # already been masked
        # calculate the difference between the expected
        if self.secondary_name is not None: 
            slice_data['ce'] = slice_data['e1']-slice_data['e2']
            # calculate the difference between the differences in raw-expected
            slice_data['cd'] = slice_data['d1']-slice_data['d2']
            
            if np.sum(slice_data['cd']) <= self.cd_filter and slice_data['y'] == 1:
                return None
            # non negative differences for the basecalls and phredq since they go to embedding layers
            slice_data['cs'] = self.non_negative_diff(slice_data['s1'].astype(int)-slice_data['s2'].astype(int), offset = 5)
            slice_data['cp'] = self.non_negative_diff(slice_data['p1'].astype(int)-slice_data['p2'].astype(int), offset = 100)

        if file_idx is not None:
            slice_data['file_idx'] = file_idx
            slice_data['sample_idx'] = idx

        return slice_data

    @staticmethod
    @jit(nopython=True)
    def non_negative_diff(d, offset):

        neg_pos = np.where(d < 0)[0] 
        d = np.abs(d)
        d[neg_pos] += offset
        return d

    @staticmethod
    @jit(nopython=True)
    def mask_data(x, mask_st, mask_nd, mask_value):
        x[:mask_st] = mask_value
        x[mask_nd:] = mask_value
        return x

    @staticmethod
    @jit(nopython=True)
    def get_mask(calls, non_masked_bases):

        # should return st and nd masking positions

        mask_st = 0
        mask_nd = calls.shape[0]
        mid_point = calls.shape[0]//2
        bpos = np.where(calls > 0)[0]
        left_bases = np.where(bpos < mid_point)[0]
        if len(left_bases) >= non_masked_bases:
            mask_st = bpos[left_bases[-non_masked_bases]]

        right_bases = np.where(bpos > mid_point)[0]
        if len(right_bases) > non_masked_bases:
            mask_nd = bpos[right_bases[non_masked_bases]]

        return mask_st, mask_nd
    
    def _get_samples_per_file(self):
        """Gets the number of samples per file from the file name
        """
        print('Getting dataset sizes')
        l = list()
        failed_files = False
        self.samples_per_file = np.zeros((len(self.files_list), ), dtype = int)

        for i, f in tqdm(enumerate(self.files_list)):
            try:
                if self.inference_mode:
                    l.append(read_metadata(os.path.join(self.data_dir, f))[0][1][0])
                else:
                    arr = np.load(os.path.join(self.data_dir, f))
                    
            except:
                print('Error reading: {}'.format(os.path.join(self.data_dir, f)))
                failed_files = True
                continue

            self.samples_per_file[i] += len(arr['y'])

        if failed_files:
            raise ValueError("Some files are corrupted")
    
   
    def _get_samplers(self, cvs = ['train', 'validation', 'test']):
        """Add samplers
        """
        self.samplers = dict()
        for cvset, neg_file_idxs, pos_file_idxs in zip(
            cvs, 
            [self.train_files_idxs, self.validation_files_idxs, self.test_files_idxs],
            [self.train_pos_file_idx, self.validation_pos_file_idx, self.test_pos_file_idx]
        ):
            self.samplers[cvset] = BalanceSampler2(
                data_source = self,
                neg_file_idxs = neg_file_idxs,
                neg_samples_per_file = self.samples_per_file[neg_file_idxs], 
                pos_file_idx = pos_file_idxs,
                n_positive_idxs = self.samples_per_file[pos_file_idxs],
                balance_mode = self.balance_mode, 
                reduce_dataset = self.reduce_dataset,
            )

        return self.samplers
    
class BalanceSampler2(Sampler):
    """Sampler that balances positives and negatives in an unbalanced dataset.
    Assumes that the positive class is underrepresented

    Args:
        neg_samples_per_file: np.array with the number of negative samples per file
        pos_samples_per_file: np.array with the number of positive samples per file
        balance_mode: 'upsample' for upsampling the positive class to the number
            of negative samples. 'downsample' for downsampling the negative class
            to the number of positive samples.
        reduce_dataset: use only a fraction of the samples
    """
    def __init__(
            self, 
            neg_file_idxs,
            neg_samples_per_file, 
            pos_file_idx,
            n_positive_idxs,
            balance_mode, 
            reduce_dataset = 1.0,
            *args, 
            **kwargs
            ):
        
        super(BalanceSampler2, self).__init__(*args, **kwargs)
        self.neg_file_idxs = neg_file_idxs
        self.neg_samples_per_file = neg_samples_per_file
        self.pos_file_idx = pos_file_idx
        self.positive_idxs = np.arange(n_positive_idxs)
        self.balance_mode = balance_mode
        self.reduce_dataset = reduce_dataset

        self.total_neg_samples = neg_samples_per_file.sum()
        self.total_files = len(self.neg_samples_per_file)
        
        if self.balance_mode == 'upsample':
            self.total_samples = int(self.total_neg_samples * 2 * self.reduce_dataset)
        elif self.balance_mode == 'downsample':
            raise NotImplementedError
            self.total_samples = self.total_pos_samples * 2
        elif self.balance_mode == 'unbalanced':
            self.total_samples = self.total_neg_samples + n_positive_idxs
            self.ratio = self.total_samples // n_positive_idxs
        else:
            raise ValueError('Unrecognized balance_mode: {}. Options: "upsample" or "downsample" or "unbalanced"'.format(balance_mode))

        self.current_file = 0
        self.current_class = 0
        self.ni = 0
        self.pi = 0
        self.i = 0
        self.neg_idxs = self.prepare_neg_idxs()
        self.pos_idxs = self.prepare_pos_idxs()

    def prepare_pos_idxs(self):
        self.pi = 0
        pos_idxs = deepcopy(self.positive_idxs)
        np.random.seed(self.i)
        np.random.shuffle(pos_idxs)
        pos_idxs = pos_idxs.astype(int)

        return pos_idxs

    def prepare_neg_idxs(self):
        self.ni = 0
        neg_idxs = np.arange(self.neg_samples_per_file[self.current_file])
        neg_idxs = neg_idxs.astype(int)
        np.random.seed(self.current_file)
        np.random.shuffle(neg_idxs)
        neg_idxs = neg_idxs[:int(len(neg_idxs) * self.reduce_dataset)]

        return neg_idxs

    def __iter__(self):
        return self

    def __len__(self):
        return self.total_samples
    
    def __next__(self):

        if self.i >= self.total_samples:
            raise StopIteration
        
        if self.balance_mode == 'upsample':
            return self.next_upsample()
        elif self.balance_mode == 'downsample':
            raise NotImplementedError
        elif self.balance_mode == 'unbalanced':
            return self.next_unbalanced()
    
    def next_unbalanced(self):

        if self.i % self.ratio == 0:
            try:
                result = (self.pos_file_idx, self.pos_idxs[self.pi])
                self.pi += 1
                
            except IndexError:
                result = (self.pos_file_idx, np.random.choice(self.pos_idxs))
                # reset idxs
                self.pos_idxs = self.prepare_pos_idxs()

            self.i += 1
            return result
        else:
            try:
                result = (self.neg_file_idxs[self.current_file], self.neg_idxs[self.ni])
                self.ni += 1
                
            except IndexError:
                result = (self.neg_file_idxs[self.current_file], np.random.choice(self.neg_idxs))
                
                self.current_file += 1
                if self.current_file >= self.total_files:
                    self.current_file = 0
                # reset idxs
                self.neg_idxs = self.prepare_neg_idxs()

            
            self.i += 1
            return result



    def next_upsample(self):
        if self.current_class:
            try:
                result = (self.pos_file_idx, self.pos_idxs[self.pi])
                self.pi += 1
                
            except IndexError:
                result = (self.pos_file_idx, np.random.choice(self.pos_idxs))
                # reset idxs
                self.pos_idxs = self.prepare_pos_idxs()

            self.i += 1
            self.current_class = 0
            
            return result
        
        if not self.current_class:
            try:
                result = (self.neg_file_idxs[self.current_file], self.neg_idxs[self.ni])
                self.ni += 1
                
            except IndexError:
                result = (self.neg_file_idxs[self.current_file], np.random.choice(self.neg_idxs))
                
                self.current_file += 1
                if self.current_file >= self.total_files:
                    self.current_file = 0
                # reset idxs
                self.neg_idxs = self.prepare_neg_idxs()

            
            self.i += 1
            self.current_class = 1
            
            return result