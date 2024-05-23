import numpy as np
from tombo import tombo_helper as th
from tombo import tombo_stats as ts
from tombo import resquiggle as tr
from tombo._default_parameters import DNA_SAMP_TYPE, MIN_EVENT_TO_SEQ_RATIO, MAX_RAW_CPTS

def seq_to_signal(seq):
    """Compute expected signal levels for a sequence from a reference model
    Args:
        seq (str): genomic seqeunce to be converted to expected signal levels
        std_ref (:class:`tombo.tombo_stats.TomboModel`): expected signal level 
            model
        rev_strand (bool): flip sequence (after extracting k-mers for expected 
            level model lookup)
        alt_ref (:class:`tombo.tombo_stats.TomboModel`): an alternative     
            expected signal level model
    Note:
        Returned expected signal levels will be trimmed compared to the passed 
        sequence based on the `std_ref.kmer_width` and `std_ref.central_pos`.
    Returns:
        Expected signal level references
        1) ref_means (`np.array::np.float64`) expected signal levels
        2) ref_sds (`np.array::np.float64`) expected signal level sds
        3) alt_means (`np.array::np.float64`) alternate expected signal levels
        4) alt_sds (`np.array::np.float64`) alternate expected signal level sds
    """
    
    seq_samp_type = th.seqSampleType(DNA_SAMP_TYPE, False)
    std_ref = ts.TomboModel(seq_samp_type=seq_samp_type)
    
    seq_kmers = [seq[i:i + std_ref.kmer_width]
                 for i in range(len(seq) - std_ref.kmer_width + 1)]

    ref_means = np.zeros((len(seq_kmers), ), dtype=float)
    for i, kmer in enumerate(seq_kmers):
        try:
            ref_means[i] = std_ref.means[kmer]
        except KeyError:
            raise ValueError(
                'Invalid sequence encountered from genome sequence: {}'.format(kmer)
            )
    
    return ref_means

def dp_global_base_assignment(read_id, read_data, genome_seq):
    
    seq_samp_type = th.seqSampleType(DNA_SAMP_TYPE, False)
    std_ref = ts.TomboModel(seq_samp_type=seq_samp_type)
    rsqgl_params = ts.load_resquiggle_parameters(seq_samp_type)

    all_raw_signal = read_data.raw
    
    align_info = th.alignInfo('insert_1', 
                              'insert_resquiggle', 
                              0, 0, 0, 0, len(genome_seq), 0)
    genome_loc = th.genomeLocation(0, '+', read_id)

    mean_q_score = 15
    start_clip_bases = None
    
    map_results = th.resquiggleResults(align_info=align_info, 
                                       genome_loc=genome_loc, 
                                       genome_seq=genome_seq,
                                       mean_q_score=mean_q_score, 
                                       start_clip_bases=start_clip_bases)
    
    map_results._replace(raw_signal = all_raw_signal)
    
    rsqgl_results = tr.resquiggle_read(map_results, 
                                       std_ref, 
                                       rsqgl_params, 
                                       all_raw_signal=all_raw_signal)
    
    return rsqgl_results

def segment_globally(read_id, read_reference, read_data):
    """Does global segmentation using tombo and choses the best alignment
    
    Args:
        read_id (str): read unique identifier
        read_reference (str): read reference sequence
        read_data (ReadData): fast5 data from the read
        
    Returns:
        Best dynamic programming alignment result
    """
    read_reference_no_mod = read_reference.replace('o', 'G')
    read_reference_no_mod = read_reference_no_mod.upper()
    
    static_res = dp_global_base_assignment(read_id, read_data, read_reference_no_mod)
    
    return static_res

def resquiggle_read_normalized(read_id, raw_signal, genome_seq, norm_signal):
    """This is an adaptation of tr.resquiggle_read to skip the normalization since 
    we are only giving a portion of the signal.
    """
    
    ## tombo default params
    outlier_thresh = None
    const_scale = None
    max_raw_cpts = MAX_RAW_CPTS
    
    seq_samp_type = th.seqSampleType(DNA_SAMP_TYPE, False)
    std_ref = ts.TomboModel(seq_samp_type=seq_samp_type)
    rsqgl_params = ts.load_resquiggle_parameters(seq_samp_type)

    all_raw_signal = raw_signal
    
    align_info = th.alignInfo('insert_1', 
                              'insert_resquiggle', 
                              0, 0, 0, 0, len(genome_seq), 0)
    genome_loc = th.genomeLocation(0, '+', read_id)

    mean_q_score = 15
    start_clip_bases = None
    
    map_res = th.resquiggleResults(align_info=align_info, 
                                   genome_loc=genome_loc, 
                                   genome_seq=genome_seq,
                                   mean_q_score=mean_q_score, 
                                   start_clip_bases=start_clip_bases)
    
    map_res = map_res._replace(raw_signal = all_raw_signal)
    
    # compute number of events to find
    # ensure at least a minimal number of events per mapped sequence are found
    num_mapped_bases = len(map_res.genome_seq) - std_ref.kmer_width + 1
    num_events = ts.compute_num_events(
        map_res.raw_signal.shape[0], num_mapped_bases,
        rsqgl_params.mean_obs_per_event, MIN_EVENT_TO_SEQ_RATIO)
    # ensure that there isn't *far* too much signal for the mapped sequence
    # i.e. one adaptive bandwidth per base is too much to find a good mapping
    if num_events / rsqgl_params.bandwidth > num_mapped_bases:
        raise th.TomboError('Too much raw signal for mapped sequence')
        
    ## here dont get the normalized signal from segment signal, use our
    ## own passed as an argument
    
    valid_cpts, _, new_scale_values = tr.segment_signal(
        map_res, num_events, rsqgl_params, outlier_thresh, const_scale)
    event_means = ts.compute_base_means(norm_signal, valid_cpts)
    
    dp_res = tr.find_adaptive_base_assignment(
        valid_cpts, event_means, rsqgl_params, std_ref, map_res.genome_seq,
        start_clip_bases=map_res.start_clip_bases,
        seq_samp_type=seq_samp_type, reg_id=map_res.align_info.ID)
    # clip raw signal to only part mapping to genome seq
    norm_signal = norm_signal[dp_res.read_start_rel_to_raw:
                              dp_res.read_start_rel_to_raw + dp_res.segs[-1]]
    
    segs = tr.resolve_skipped_bases_with_raw(
        dp_res, norm_signal, rsqgl_params, max_raw_cpts)
    
    norm_params_changed = False
    
    sig_match_score = ts.get_read_seg_score(
        ts.compute_base_means(norm_signal, segs),
        dp_res.ref_means, dp_res.ref_sds)
    if segs.shape[0] != len(dp_res.genome_seq) + 1:
        raise th.TomboError('Aligned sequence does not match number ' +
                            'of segments produced')

    return map_res._replace(
        read_start_rel_to_raw=dp_res.read_start_rel_to_raw, segs=segs,
        genome_seq=dp_res.genome_seq, raw_signal=norm_signal,
        scale_values=new_scale_values, sig_match_score=sig_match_score,
        norm_params_changed=norm_params_changed)