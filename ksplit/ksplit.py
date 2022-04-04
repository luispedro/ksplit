import pyximport
import numpy as np
pyximport.install(setup_args={
    'include_dirs': np.get_include()
    })

from . import kmers
from .fastq import fastq_iter
from .fasta import fasta_iter

def encode_fastq(args, ifile, kmer_size : int, out):
    '''
    Returns the number of sequences encoded


    Check the file `ALGORITHM.md` for a description of the on-disk format

    Parameters
    ----------

    args : arguments
    ifile (file-like): input file object (should be a FastQ file)
    kmer_size : int
        K-mer size
    ofile (file-like): output file object (should be opened in binary mode)

    Example
    -------
    encode_fastq( { .. }, open(ifname, 'rb'), open(ofname, 'wb'))
    '''
    for i,seqs in enumerate(fastq_iter(ifile)):
        ks = []
        for seq in seqs:
            k = kmers.kmers(seq.seq, kmer_size)
            if k is None:
                raise ValueError("Something wrong!")
            ks.append(k)

        if len(ks) == 1:
            [ks] = ks
        else:
            ks = np.concatenate(ks)
        encoded = np.empty((len(ks), 2), dtype=np.uint64)
        encoded.T[0] = ks
        encoded.T[1] = i
        out.write(encoded.data)
    return i + 1


def encode_fasta(args, input_file, kmer_size : int, output_file):
    '''
    Encode all the sequences in the input (FASTA) file to disk.

    Check the file `ALGORITHM.md` for a description of the on-disk format

    Parameters
    ----------

    args : dict
        Unused
    input_file (file-like): input file object (should be a FASTA file)
    kmer_size : int
        K-mer size
    output_file (file-like): output file object (should be opened in binary mode)

    Example
    -------
    encode_fasta( { .. }, open(input_file_name, 'rt'), kmer_size, open(output_file_name, 'wb'))

    Returns
    -------
    n : int,
        The number of sequences encoded
    '''

    for i, (_, seq) in enumerate(fasta_iter(input_file)):
        kmers_array = kmers.kmers(seq.encode('ascii'), kmer_size)

        if kmers_array is None:
            raise ValueError("Something wrong with sequence.",
                             "Faulty sequence is: {}.".format(seq.seq),
                             "Error at sequence #{} of input file.".format(i))

        #format and write to output file
        encoded = np.empty((len(kmers_array), 2), dtype=np.uint64)
        encoded.T[0] = kmers_array
        encoded.T[1] = i
        output_file.write(encoded.data)

    return i+1

