from io import BytesIO
import ksplit.ksplit
import numpy as np
from os import path

BASEDIR = path.join(path.dirname(path.abspath(__file__)), 'data')

fq = '''@ReadP
TATATATATATTTCTTGTAATTTGTTGGAATACGAGAACATCGTCAATAATATATCGTATGAATTGAACCACACGGCACA
+
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
'''

KMER_SIZE = 31

def test_encode_fastq():
    out = BytesIO()
    ksplit.ksplit.encode_fastq({}, BytesIO(fq.encode('ascii')), KMER_SIZE, out)
    assert set(np.frombuffer(out.getvalue(), np.uint64).reshape((-1, 2)).T[1]) == set([0])

    out = BytesIO()
    with open(path.join(BASEDIR, 'interleaved.fq'), 'rb') as ifile:
        ksplit.ksplit.encode_fastq({}, ifile, KMER_SIZE, out)
    assert set(np.frombuffer(out.getvalue(), np.uint64).reshape((-1, 2)).T[1]) == set([0, 1, 2, 3])
