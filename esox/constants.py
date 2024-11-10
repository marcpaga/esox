import os

## vocabulary of the 4 canonical bases and 8-oxoG
ALL_BASES = ['A', 'C', 'G', 'T', 'o']

try:
    import parasail
    # during training accuracy alignment constants
    ACC_ALIGNMENT_GAP_OPEN_PENALTY = 8
    ACC_ALIGNMENT_GAP_EXTEND_PENALTY = 4
    ACC_ALIGNMENT_MATCH_SCORE = 2
    ACC_ALIGNMENT_MISSMATCH_SCORE = 1
    ACC_GLOBAL_ALIGN_FUNCTION = parasail.nw_trace_striped_32
    ACC_MATRIX = parasail.matrix_create("".join("".join(ALL_BASES)), ACC_ALIGNMENT_MATCH_SCORE, -ACC_ALIGNMENT_MISSMATCH_SCORE)

    # oligo repeat finding alignment constants
    ALIGNMENT_MATCH_SCORE = 3
    ALIGNMENT_MISSMATCH_SCORE = 3
    GAP_OPEN_PENALTY = 8
    GAP_EXTEND_PENALTY = 4
    LOCAL_ALIGN_FUNCTION = parasail.sg_qx_trace_striped_16
    MATRIX = parasail.matrix_from_filename = parasail.Matrix(os.path.join(os.path.abspath(os.path.dirname(__file__)), "oligo_matrix.txt"))
except ImportError:
    print('No alignment library')

## design of the oligos
# B = barcode
# N = random base
# M = modified bases
# K = kmer bases
# H = overhang bases
FWD_OLIGO_DESIGN = 'BBBBBBBNNMMMMMNNNBBBBBBBKKKKKBBBBBBBHHHHHHHHHH'
REV_OLIGO_DESIGN = 'BBBBBBBKKKKKBBBBBBBNNNKKKKKNNBBBBBBBHHHHHHHHHH'


HEAD_ADAPTER = 'AATGTACTTCGTTCAGTTACGTATTGCT'
TAIL_ADAPTER = 'GCAATACGTAACTGAACGAAGT'

PHRED_NUMS={}
for x in range(0,94):
    PHRED_NUMS.update({x : (chr(x+33).encode('ascii'))})
PHRED_LETTERS = dict()
for k, v in PHRED_NUMS.items():
    PHRED_LETTERS[v.decode()] = k

POS_PHRED = "".join([str(v.decode()) for v in PHRED_NUMS.values()])

CRF_STATE_LEN = 4
CRF_BIAS = True
CRF_SCALE = 5.0
CRF_BLANK_SCORE = 2.0
BLANK_TOKEN = 0

ENCODING_DICT_CRF = {
    "" : 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "o": 5,
    "a": 1,
    "c": 2,
    "g": 3,
    "t": 4,
}
DECODING_DICT_CRF = {
    0:  "",
    1: "A",
    2: "C",
    3: "G",
    4: "T",
    5: "o",
}