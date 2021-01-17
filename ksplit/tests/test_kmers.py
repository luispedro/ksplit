from hypothesis import given, note, strategies as st
import pyximport
import numpy as np
pyximport.install(setup_args={
    'include_dirs': np.get_include()
    })

from ksplit import kmers

KMER_SIZE = 31

def test_kmers():
    testing = 'ATTTAACATGAGATAACATGCATGCATGCATTGCGGCTCAGCTAGTCAGCTAGCTAGCTAGCTACGATCGATCGTAGCATCGATCGATCGATCGATCGATCGATCGTACGTACGTAGCTACGATCGTAGCTAGCTAG'

    testing = testing.encode('ascii')
    print(kmers.kmers(testing, KMER_SIZE))

    testing = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'

    testing = testing.encode('ascii')
    print(kmers.kmers(testing, KMER_SIZE))


def test_error_detection():
    testing = 'ATTTAACATGAGXTAACATGCATGCATGCAT'
    testing = testing.encode('ascii')
    assert kmers.kmers(testing, KMER_SIZE) is None


def test_kmers1():
    testing = 'TTTCTTTTTTTTTTTTTTTTTTTTTTTTTTT'
    testing = testing.encode('ascii')
    ks = kmers.kmers(testing, KMER_SIZE)
    assert len(testing) == KMER_SIZE
    assert len(ks) == 1

def rc(t):
    rcd = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
    }
    return ''.join([rcd[c] for c in t[::-1]])


def test_kmers_reverse():
    for t in [
            'TTATACATACTGTTGGTATGATAATAGTATA',
            'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT',
            'ATTTTTTTTTTTTTTTTTTTTTTTTTTTTTT',
            'AAAAAAAAAAATTTTTTTTTTTTTTTTTTTT',
            ]:
        assert np.all(kmers.kmers(t.encode('ascii'), KMER_SIZE)
                    == kmers.kmers(rc(t).encode('ascii'), KMER_SIZE))


def test_kmers_reverse_embed():
    k = 'TTATACATACTGTTGGTATGATAATAGTATA'

    t0 = k + 'C'
    t1 = rc(k) + 'T'
    assert kmers.kmers(t0.encode('ascii'), KMER_SIZE)[0] == kmers.kmers(t1.encode('ascii'), KMER_SIZE)[0]

def test_max62():
    assert len('{:b}'.format(kmers.kmers('GACATAGCGACGCGGACCCCCTTTTTTTTTTGG'.encode('ascii'), KMER_SIZE).max())) <= 62

def test_kmers_different():
    ks = [
        'TTATACATACTGTTGGTATGATAATAGTATA',
        'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT',
        'ATTTTTTTTTTTTTTTTTTTTTTTTTTTTTT',
        'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTA',
        ]
    ks = [kmers.kmers(k.encode('ascii'), KMER_SIZE)[0] for k in ks]
    assert len(ks) == len(set(ks))

def test_regression_kmers():
    "regression on kmer computation"
    ks_c = kmers.kmers(b'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC', KMER_SIZE)
    ks_g = kmers.kmers(b'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG', KMER_SIZE)
    assert len(set(ks_c)) == 1
    assert len(set(ks_g)) == 1
    assert np.all(ks_c == ks_g)



def  encode_nt(nt):
    if type(nt) == str:
        nt = nt.encode('ascii')
    if nt == b'A': return 0
    if nt == b'C': return 1
    if nt == b'T': return 2
    if nt == b'G': return 3
    return -1

def rc1(nt):
    return {
            'A': 'T',
            'T': 'A',
            'C': 'G',
            'G' : 'C'
            }[nt]
def rc(s):
    return ''.join(rc1(si) for si in s[::-1])

def encode_kmer(k):
    return int(''.join(['{:02b}'.format(encode_nt(ki)) for ki in k[::-1]]),2)

def encode_kmer_rc(k):
    return encode_kmer(rc(k))

def encode_kmer_min(k):
    assert len(k) == KMER_SIZE
    return min(encode_kmer(k),
                encode_kmer_rc(k))

@given(seq=st.text(alphabet='ATGC', min_size=KMER_SIZE, max_size=65))
def test_naive(seq):
    import numpy as np
    n = np.array([encode_kmer_min(seq[i:i+KMER_SIZE])
                            for i in range(len(seq) - KMER_SIZE + 1)])
    fast = kmers.kmers(seq.encode('ascii'), KMER_SIZE)
    assert len(n) == len(fast)
    assert np.all(n == fast)

@given(seq=st.text(alphabet='ATGC', min_size=KMER_SIZE, max_size=65))
def test_shift(seq):
    import numpy as np
    shifted = np.array([kmers.kmers(seq[i:i+KMER_SIZE].encode('ascii'), KMER_SIZE)[0]
                            for i in range(len(seq) - KMER_SIZE + 1)])
    fast = kmers.kmers(seq.encode('ascii'), KMER_SIZE)
    assert len(shifted) == len(fast)
    assert np.all(shifted == fast)



