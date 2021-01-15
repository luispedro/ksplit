from Bio import SeqIO
import pyximport
import numpy as np
pyximport.install(setup_args={
    'include_dirs': np.get_include()
    })

from . import kmers
from .fastq import fastq_iter

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
            # Below is a major hack, but this was actually done by SOAP too
            seq = seq.seq.replace(b'N', b'A')
            k = kmers.kmers(seq, kmer_size)
            if k is None:
                raise ValueError("Something wrong!")
            ks.append(k)

        # TODO This reorganization of the ks could be done in one step without
        # generating the intermediate arrays, but it would require somewhat
        # more complex code:
        if len(ks) == 1:
            [ks] = ks
        else:
            ks = np.concatenate(ks)
        ixs = np.repeat(np.array([i], dtype=np.uint64), len(ks))
        encoded = np.vstack([ks, ixs]).T.ravel()
        out.write(encoded.data)
    return i + 1


def encode_fasta(args, input_file, kmer_size, output_file):
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
    encode_fasta( { .. }, open(input_file_name, 'rt'), open(output_file_name, 'wb'))

    Returns
    -------
    n : int,
        The number of sequences encoded
    '''

    for i, seq in enumerate(SeqIO.parse(input_file, 'fasta')):
        #encode sequence, then compute kmers for that sequence
        ascii_encoded_sequence = seq.seq.encode('ascii')
        kmers_array = kmers.kmers(ascii_encoded_sequence, kmer_size)

        #checks if current sequence has a character other than ACTG
        if kmers_array is None:

            #updates read according to IUPAC rules
            ascii_encoded_sequence =\
            ascii_encoded_sequence.replace(b'N', b'A').replace(b'M', b'A').\
            replace(b'R', b'A').replace(b'K', b'G').replace(b'Y', b'C').\
            replace(b'S',b'G').replace(b'W',b'A').replace(b'B',b'C').\
            replace(b'D',b'A').replace(b'H',b'A').replace(b'V',b'A')

            #recomputes kmers for the sequence
            kmers_array = kmers.kmers(ascii_encoded_sequence, kmer_size)

            if kmers_array is None:
                raise ValueError("Something wrong with sequence.",
                                 "Faulty sequence is: {}.".format(ascii_encoded_sequence),
                                 "Error at iteration {} of input file.".format(i))

        #format and write to output file
        ixs = np.repeat(np.array([i], dtype=np.uint64), len(kmers_array))
        encoded = np.vstack([kmers_array, ixs]).T.ravel()
        output_file.write(encoded.data)

    return i+1

