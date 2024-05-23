"""Functions for data normalization
"""

import numpy as np
from scipy.signal import find_peaks

from esox.resquiggle import resquiggle_read_normalized, seq_to_signal

## normalize the signal
def med_mad(signal, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    
    Args:
        signal (np.array): array of data to calculate med and mad
        factor (float): factor to scale the mad
        
    Returns:
        float, float : med and mad values
    """
    med = np.median(signal)
    mad = np.median(np.absolute(signal - med)) * factor
    return med, mad


def find_noisiest_section(signal, samples=100, threshold=6.0):
    """Find the noisiest section of a signal.
    
    Args:
        signal (np.array): raw nanopore signal
        samples (int): defaults to 100
        threshold (float): defaults to 6.0
        
    Returns:
        np.array : with a section (or all) the input signal that has the noisiest section
    """
    
    threshold = signal.std() / threshold
    noise = np.ones(signal.shape)

    for idx in np.arange(signal.shape[0] // samples):
        window = slice(idx * samples, (idx + 1) * samples)
        noise[window] = np.where(signal[window].std() > threshold, 1, 0)

    # start and end low for peak finding
    noise[0] = 0; noise[-1] = 0
    peaks, info = find_peaks(noise, width=(None, None))

    if len(peaks):
        widest = np.argmax(info['widths'])
        tonorm = signal[info['left_bases'][widest]: info['right_bases'][widest]]
    else:
        tonorm = signal
        
    return tonorm

def scale_signal(signal, offset, range, digitisation):
    """Scale the signal to pA as explained in taiyaki
    
    Args:
        signal (np.array): raw signal to be normalized
        offset (int): offset as indicated in the attributes of the read fast5 file
        range (float): range as indicated in the attributes of the read fast5 file
        digitisation (float): as indicated in the attributes of the read fast5 file
        
    Returns:
        np.array : scaled signal
    """

    scaled_signal = (signal + offset) * range/digitisation
    
    return scaled_signal

def rescale_lstsq(dacs, levels, med, mad):
    norm_sig = (dacs - med) / mad
    shift_est, scale_est = np.linalg.lstsq(
        np.column_stack([np.ones_like(norm_sig), norm_sig]),
        levels,
        rcond=None,
    )[0]
    new_shift = med - (mad * shift_est / scale_est)
    new_scale = mad / scale_est
    return new_shift, new_scale

def normalize_signal(signal, med, mad):
    """ Rescale a signal based on the med and mad
    
    The signal is median centered and mad scaled.
    
    Args:
        signal (np.array): signal to be rescaled
        med (float): median to be substracted
        mad (float): median absolute deviation to be used for scaling
        
    Returns:
        np.array with the normalized signal
    """
    
    signal = signal.astype(np.float32)
    signal -= med
    signal /= mad
    return signal

def normalize_signal_wrapper(signal, offset, range, digitisation, method = 'noisiest', samples=100, threshold=6.0, factor=1.4826):
    """Wrapper function to normalize the nanopore raw signal
    
    Args:
        signal (np.array): raw signal to be normalized
        offset (int): offset as indicated in the attributes of the read fast5 file
        range (float): range as indicated in the attributes of the read fast5 file
        digitisation (float): as indicated in the attributes of the read fast5 file
        method (str): how to define the range of values to use to calculate the
            med and the mad. Can be "noisiest" (searches for the noisiest part
            in the signal) or "all" (uses all the signal).
        samples (int): used to find noisiest section, defaults to 100
        threshold (float): used to find the noisiest section, defaults to 6.0
        factor (float): mad scaler, defaults to 1.4826
        
    Returns:
        np.array: normalized signal
    """
    
    scaled_signal = scale_signal(signal, offset = offset, range = range, digitisation = digitisation)
    
    if method == 'noisiest':
        med_mad_signal = find_noisiest_section(scaled_signal, 
                                               samples=samples, 
                                               threshold=threshold)
    elif method == 'all':
        med_mad_signal = scaled_signal
    else:
        raise ValueError('Method should be "noisiest" or "all"')
    
    med, mad = med_mad(med_mad_signal, factor = factor)
    
    normalized_signal = normalize_signal(scaled_signal, med = med, mad = mad)
    return normalized_signal

def normalize_signal_from_read_data(read_data):
    """Normalize the nanopore raw signal
    
    Args:
        read_data (ReadData)
    """
    
    return normalize_signal_wrapper(read_data.raw, offset = read_data.offset, range = read_data.range, digitisation = read_data.digitisation)

def rescale_data(fast5_data, basecalls, read_id):
    """
    Args:
        fast5_data (ReadData): a ReadData object with fast5 data
        basecalls (str): basecalls or true sequence for that sequence
        read_id (str): read_id

    Returns:
        Rescaled raw signal according to tombo expected signal
    """

    raw_signal = fast5_data.raw
    med, mad = med_mad(raw_signal, factor = 1)
    norm_signal = (raw_signal - med)/mad

    resquiggle_results = resquiggle_read_normalized(
        read_id = read_id, 
        raw_signal = raw_signal, 
        genome_seq = basecalls, 
        norm_signal = norm_signal,
    )

    seg_table = resquiggle_results.segs
    expected_signal = seq_to_signal(basecalls)
    segment_lengths = seg_table[1:] - seg_table[:-1]

    expected_long_signal = np.zeros((np.sum(segment_lengths),), dtype = float)
    c = 0
    for sl, es in zip(segment_lengths, expected_signal):

        expected_long_signal[c:sl+c] = es
        c += sl


    rescale_st = resquiggle_results.read_start_rel_to_raw
    rescale_nd = rescale_st + len(expected_long_signal)
    raw_rescale = fast5_data.raw[rescale_st:rescale_nd]

    new_med, new_mad = rescale_lstsq(raw_rescale, expected_long_signal, med, mad)

    rescaled_data = (fast5_data.raw - new_med) / new_mad
    return rescaled_data