# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

ENCODING = {"A": [1, 0, 0, 0],
            "T": [0, 1, 0, 0],
            "C": [0, 0, 1, 0],
            "G": [0, 0, 0, 1]}
ALPHABET = set(ENCODING.keys())

# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """



    seq_enc = []
    for seq in [s.upper() for s in seq_arr]:
        if len(set(seq) - ALPHABET) > 0: # check if any non ATCG characters
            raise ValueError(f"There is a character in the passed sequence {seq} that is not in the alphabet {ALPHABET}.")

        encoded = []
        for l in seq:
            encoded += ENCODING[l]
        seq_enc += [encoded]

    return seq_enc


def sample_seqs(
        seqs: List[str],
        labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance.
    Consider this as a sampling scheme with replacement.

    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    n = len(labels)
    positive_idx = list(np.where(np.array(labels)==1)[0]) # all indices of seqs/labels corresponding to True
    negative_idx = list(np.where(np.array(labels)==0)[0]) # all indices of seqs/labels corresponding to False


    sampled_seqs = []
    sampled_labels = []
    for i in range(len(positive_idx)*2):
        coin_flip = np.random.uniform()
        if coin_flip < 0.5: # choose from positive samples with probaility 1/2
            sample = int(np.random.uniform(0, len(positive_idx))) # select a datapoint at random
            sampled_seqs.append(seqs[positive_idx[sample]])
            sampled_labels.append(labels[positive_idx[sample]])
        else: # choose from negative samples with probability 1/2
            sample = int(np.random.uniform(0, len(negative_idx))) # select a datapoint at random
            sampled_seqs.append(seqs[negative_idx[sample]])
            sampled_labels.append(labels[negative_idx[sample]])

    return sampled_seqs, sampled_labels

def read_fasta(path: str) -> List[str]:
    """
    Given a path to a fa file, read it in line by line and return sequences as a list of strings. 
    
    Args:
        path: str
            Path to .fa file 
    
    Returns:
        seqs: List[str]
            Sequences stored in a list of strings 
    """
    seqs = []
    seq = ""
    with open(path) as file:
        for line in file:
            if ">" in line: # sequence header, can skip
                seqs.append(seq) # keep track of full sequence
                seq = ""
            else:
                seq += line.strip("\n")
    return seqs[1:] # the first entry will just be \"\", can remove this 