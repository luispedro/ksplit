import numpy as np
from io import BytesIO
from ksplit import ksplit, sort
import numpy as np
from os import stat

fq = '''@ReadP
TATATATATATTTCTTGTAATTTGTTGGAATACGAGAACATCGTCAATAATATATCGTATGAATTGAACCACACGGCACA
+
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
'''

KMER_SIZE = 31

def _encoded(fq):
    out = BytesIO()
    ksplit.encode_fastq({}, BytesIO(fq.encode('ascii')), KMER_SIZE, out)
    return out.getvalue()

def _is_sorted(arr):
    c = arr.copy()
    c.sort()
    return np.all(c == arr)


def test_sort_partials(tmpdir):
    import operator
    import functools

    data = _encoded(fq)
    full = sort._from_buffer(data)
    sp = sort.sort_partials(BytesIO(data), tmpdir, block_nbytes=16*8*2)
    assert sum(stat(f).st_size for f in sp) == len(data)

    blocks = [sort._from_buffer(open(f, 'rb').read()) for f in sp]
    assert functools.reduce(operator.or_, (set(block.T[0]) for block in blocks)) == set(full.T[0])

    for b in blocks:
        assert _is_sorted(b.T[0])

def test_merge(tmpdir):
    out = BytesIO()
    data = _encoded(fq)
    full = sort._from_buffer(data)
    sp = sort.sort_partials(BytesIO(data), tmpdir, block_nbytes=16*8*2)
    out = BytesIO()
    bufs = [sort.file_buffer(f) for f in sp]
    sort.merge_streams(bufs, out, 128)
    full_s = sort._from_buffer(out.getvalue())

    assert _is_sorted(full_s.T[0])
    assert set(full_s.T[0]) == set(full.T[0])
    assert full_s.shape == full.shape

