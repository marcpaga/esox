"""
Functions and classes to read data.
"""
import os
from ont_fast5_api.fast5_interface import get_fast5_file

class ReadData:

    """Contains all the data for a Nanopore read
    
    Attributes:
        read_id (str): id of the read
        run_id (str): id of the run
        filename (str): filename where the read is stored
        offset (int): offset of the read signal
        sampling_rate (float): sampling rate of the run
        scaling (float): range/digitisation
        range (float): range of the raw data
        digitisation (float): digitisation
        mux (int): ?
        channel (str): pore channel
        start (float): time when the read was started to sequence, in seconds
        duration (float): time it took to sequence the read, in seconds
        template_start (float): time when the read was started to sequence, in seconds
        template_duration (float): time it took to sequence the read, in seconds
        raw (np.array::int): raw signal
        scaled (np.array::float): scaled signal
        signal (np.array::float): normalized scaled signal
        basecalls (str): basecalls of the read, optional
        phredq (str): quality score of the basecalls, optional
        segmentation (np.array::object): info regarding the segments of the read by tombo
        start_rel_to_raw (int): datapoint where the aligned read starts
        alignment_info (dict): info regarding the alignment that tombo made
        
    """
    
    def __init__(self, read, filename):

        self.read_id = read.read_id
        try:
            self.run_id = read.get_run_id().decode()
        except AttributeError:
            self.run_id = read.get_run_id()
        self.filename = os.path.basename(read.filename)

        read_attrs = read.handle[read.raw_dataset_group_name].attrs
        channel_info = read.handle[read.global_key + 'channel_id'].attrs

        ## signal normalization stuff
        self.offset = int(channel_info['offset'])
        self.sampling_rate = channel_info['sampling_rate']
        self.scaling = channel_info['range'] / channel_info['digitisation']
        self.range = channel_info['range']
        self.digitisation = channel_info['digitisation']

        ## channel stuff
        self.mux = read_attrs['start_mux']
        try:
            self.channel = channel_info['channel_number'].decode()
        except AttributeError:
            self.channel = channel_info['channel_number']
        self.start = read_attrs['start_time'] / self.sampling_rate
        self.duration = read_attrs['duration'] / self.sampling_rate

        # no trimming
        self.template_start = self.start
        self.template_duration = self.duration
        
        ## signal part
        self.raw = read.handle[read.raw_dataset_name][:]
        
        ## if there are basecalls
        try:
            self.basecalls = read.get_analysis_dataset('Basecall_1D_000', 'BaseCalled_template/Fastq').split('\n')[1]
            self.phredq = read.get_analysis_dataset('Basecall_1D_000', 'BaseCalled_template/Fastq').split('\n')[-2]
        except:
            self.basecalls = None
            self.phredq = None
        
        ## if there is tombo resquiggle info
        try:
            ## this succeeds if the data is put by tombo
            self.segmentation = read.get_analysis_dataset('RawGenomeCorrected_000', 'BaseCalled_template/Events')
            self.start_rel_to_raw = read.get_analysis_attributes('RawGenomeCorrected_000/BaseCalled_template/Events')['read_start_rel_to_raw']
            self.alignment_info = read.get_analysis_attributes('RawGenomeCorrected_000/BaseCalled_template/Alignment')
        except TypeError:
            try:
                ## if the data we put it ourselves we use this
                self.segmentation = read.handle['RawGenomeCorrected_000']['BaseCalled_template/Events'][:]
                self.start_rel_to_raw = read.handle['RawGenomeCorrected_000']['BaseCalled_template/Events'].attrs['read_start_rel_to_raw']
                self.alignment_info = None
            except KeyError:
                self.segmentation = None
                self.start_rel_to_raw = None
                self.alignment_info = None
            
    
    def is_basecalled(self):
        if self.basecalls is None:
            return False
        else:
            return True
    
    def is_resquiggled(self):
        if self.segmentation is None:
            return False
        else:
            return True



def list_reads_ids(fast5_file):
    """Get the available read ids from a fast5 file

    Args:
    	fast5_file (str): fast5 file that we want the ids of

    Returns:
    	A list with the read ids in the file.
    """
    
    return(get_fast5_file(fast5_file, 'r').get_read_ids())

def handle_fast5_file(fast5_file):
    """Open a fast5 file with the ONT API
    """
    
    f5_handler = get_fast5_file(fast5_file, 'r')
    
    if len(f5_handler.get_read_ids()) < 1:
        raise NameError('This file does not contain read ids')
        
    return f5_handler
    
def read_fast5(fast5_file, read_ids = None):
    """Extract the data from a fast5 file from the desired reads

    Args:
    	fast5_file (str): file to be read

    Returns:
    	A dictionary with read ids as keys and ReadData objects as
    		values.
    """
    
    read_reads = dict()
    with get_fast5_file(fast5_file, 'r') as f5_fh:
        if read_ids is None:
            for read in f5_fh.get_reads():
                read_reads[read.read_id] = (ReadData(read, fast5_file))
        else:
            if not isinstance(read_ids, list):
                read_ids = [read_ids]
            for read_id in read_ids:
                read = f5_fh.get_read(read_id)
                read_reads[read.read_id] = (ReadData(read, fast5_file))
        
    return read_reads

def iter_fasta(fasta_file):
    """Read a fasta file iteratively
    """
    c = 0
    with open(fasta_file, 'r') as handle:
        for line in handle:
            if c == 0:
                read_id = line[1:].strip('\n')
                c += 1
            elif c == 1:
                seq = line.strip('\n')
                c = 0
                yield read_id, seq

def read_fasta(fasta_file):
    """Read a fasta file
    """
        
    fasta_dict = dict()
    
    for k, v in iter_fasta(fasta_file):
        fasta_dict[k] = v

    return fasta_dict

def iter_fastq(fastq_file, validate):

    c = 0
    with open(fastq_file, 'r') as f:
        for line in f:
            if c == 0:
                c += 1
                read_id = line.split(' ')[0][1:].strip('\n')
            elif c == 1:
                c += 1
                seq = line.strip('\n')
            elif c == 2:
                c += 1
                direction = line.strip('\n')
            elif c == 3:
                c = 0
                phredq = line.strip('\n')
                if validate:
                    if len(seq) != len(phredq):
                        raise ValueError('{}: seq ({}) and phredq ({}) lenghts are different'.format(read_id, len(seq), len(phredq)))
                yield read_id, (seq, direction, phredq)

def read_fastq(fastq_file, validate = True):
    """Read a fastq file
    """
    
    fastq_dict = dict()
    for k, v in iter_fastq(fastq_file, validate):
        fastq_dict[k] = v
                
    return fastq_dict

def read_fast(fast_file, validate = True):

    if fast_file.endswith('.fastq'):
        return read_fastq(fast_file, validate)
    if fast_file.endswith('.fasta'):
        return read_fasta(fast_file)
    if fast_file.endswith('.fna') or fast_file.endswith('.fa'):
        return read_fna(fast_file)


                
def read_fna(file):
    """Read a fna file, like fasta but sequences are split by \n 
    """
    d = dict()
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                k = line.strip('\n')
                d[k[1:]] = list()
            else:
                d[k[1:]].append(line.strip('\n'))
                
    for k, v in d.items():
        d[k] = "".join(v)
    return d