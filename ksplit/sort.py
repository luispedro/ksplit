from os import path, makedirs
import numpy as np

class file_buffer:
    '''file object wrapper with cheap peek()ing'''
    def __init__(self, fname):
        if fname.endswith('.gz'):
            import gzip
            self.handle = gzip.open(fname, 'rb')
        else:
            self.handle = open(fname, 'rb')
        self.buf = b''


    def close(self):
        self.handle.close()


    def __del__(self):
        self.close()


    def peek(self, nbytes):
        if nbytes > len(self.buf):
            read_size = max(4096, nbytes - len(self.buf))
            self.buf += self.handle.read(read_size)
        return self.buf[:nbytes]


    def consume(self, nbytes):
        self.buf = self.buf[nbytes:]


def _from_buffer(data):
    return np.frombuffer(data, np.uint64).reshape((-1, 2))

def _read_blocks(ifile, nbytes):
    while True:
        data = ifile.read(nbytes)
        if not data:
            return
        yield _from_buffer(data)

def sort_partials(encoded_stream, tdir, *, block_nbytes=1024*1024*1024):
    '''Sort partial files
    '''
    import gzip
    splits_dir = path.join(tdir, 'splits')
    makedirs(splits_dir)
    partials = []
    for block in _read_blocks(encoded_stream, block_nbytes):
        ofname = path.join(splits_dir, 'split_{:02}.ks.gz'.format(len(partials)))
        partials.append(ofname)
        with gzip.open(ofname, 'wb', compresslevel=1) as out:
            block = block[np.argsort(block.T[0])]
            out.write(block.data)
    return partials


def merge_streams(bufs, out, block_nbytes):
    '''
    Merge sorted streams

    Parameters
    ----------
    bufs : list of file_buffer
    out : file-like (for output)
    block_nbytes : int
    '''
    # ALGORITHM
    #
    # At each step, we want to read at least block_nbytes from one of the
    # buffers. So, at each step, we first peek into the buffers and find the
    # one where the corresponding kmer value is lowest. Then, we read from all
    # the buffers up to that point (remember that they are all pre-sorted).
    # Now, we merge those chunks and output.
    #
    # Therefore, each iteration will make at least `block_nbytes` worth of
    # progress, but maybe more.

    while bufs:
        min_k = np.uint64(-1)
        # For the first pass, we could, in principle, just seek() to the right
        # position and read those 8 Bytes
        for b in bufs:
            ch = _from_buffer(b.peek(block_nbytes))
            if not ch.size:
                continue
            cur = ch.T[0][-1]
            if cur < min_k: min_k = cur
        nbufs = []
        cur = []
        for b in bufs:
            ch = _from_buffer(b.peek(block_nbytes))
            if not ch.size:
                continue
            cur_c = np.searchsorted(ch.T[0], min_k, 'right')
            b.consume(cur_c * 8 * 2)
            cur.append(ch[:cur_c])
            nbufs.append(b)
        bufs = nbufs
        if cur:
            cur = np.concatenate(cur)
            cur = cur[np.argsort(cur.T[0])]
            out.write(cur.data)

def sort_kmer_pairs(args : dict, enc, out):
    '''
    Sort kmer pairs

    Parameters
    ----------
    args : dictionary
        Arguments
            block_nbytes : int
            tempdir : str
            verbose : bool, optional
    enc : file-like
        Encoded file
    out : file-like
        Output file
    '''

    import tempfile
    block_nbytes = args['block_mbytes'] * 1024 * 1024

    with tempfile.TemporaryDirectory(dir=args.get('tempdir')) as tdir:
        sp = sort_partials(enc, tdir, block_nbytes=block_nbytes)
        if args.get('verbose'):
            print(f'Sort step 1 (of 2) finished')

        bufs = [file_buffer(f) for f in sp]
        block_mbytes_per_block = 1 + args['block_mbytes']//len(sp)
        merge_streams(bufs, out, block_nbytes=1024*1024*block_mbytes_per_block)
        if args.get('verbose'):
            print(f'Sort finished')
