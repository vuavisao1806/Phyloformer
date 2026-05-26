from itertools import combinations

import dendropy
import torch
from torch.utils.data import Dataset

ALPHABET = b"ATGC-"
LOOKUP = {char: index for index, char in enumerate(ALPHABET)}


def _encode_sequence(seq_bytes):
    """One-hot encode a DNA byte sequence; unknown characters mapped to gap (-)."""
    gap_idx = LOOKUP[ord(b"-")]
    return [LOOKUP.get(char, gap_idx) for char in seq_bytes]


def load_alignment(filepath):
    """
    Reads a FASTA or PHYLIP sequential alignment and returns a one-hot encoded
    tensor of the MSA and the corresponding taxa label order.
    """
    if filepath.endswith(".phy"):
        return _load_alignment_phy(filepath)
    return _load_alignment_fasta(filepath)


def _load_alignment_fasta(filepath):
    sequences, ids = [], []
    with open(filepath, "rb") as aln:
        for line in aln:
            line = line.strip()
            if line.startswith(b">"):
                ids.append(line[1:].decode("utf8"))
                sequences.append([])
            else:
                sequences[-1].extend(_encode_sequence(line))

    seqs = torch.tensor(sequences)
    seqs = torch.nn.functional.one_hot(seqs, num_classes=len(ALPHABET)).permute(2, 1, 0)
    return seqs, ids


def _load_alignment_phy(filepath):
    """
    Reads a PHYLIP sequential format alignment.
    First line: n_taxa seq_len
    Subsequent lines: taxon_name whitespace sequence
    """
    sequences, ids = [], []
    with open(filepath, "rb") as aln:
        lines = aln.readlines()

    # Skip header line (n_taxa seq_len)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        ids.append(parts[0].decode("utf8"))
        sequences.append(_encode_sequence(b"".join(parts[1:])))

    seqs = torch.tensor(sequences)
    seqs = torch.nn.functional.one_hot(seqs, num_classes=len(ALPHABET)).permute(2, 1, 0)
    return seqs, ids


def load_distance_matrix(filepath, ids):
    """
    Reads a newick formatted tree and returns a vector of the
    upper triangle of the corresponding pairwise distance matrix.
    The order of taxa in the rows and columns of the corresponding
    distance matrix is given by the `ids` input list.
    """

    distances = []

    with open(filepath, "r") as treefile:
        tree = dendropy.Tree.get(file=treefile, schema="newick")
    taxa = tree.taxon_namespace
    dm = tree.phylogenetic_distance_matrix()
    for tip1, tip2 in combinations(ids, 2):
        l1, l2 = taxa.get_taxon(tip1), taxa.get_taxon(tip2)
        distances.append(dm.distance(l1, l2))

    return torch.tensor(distances)


class PhyloDataset(Dataset):
    """
    Simple pytorch dataset that reads tree/alignment pairs
    and returns the corresponding tensor objects
    """

    def __init__(self, pairs):
        """
        pairs: List[(str,str)] = a list of (treefile, alnfile) paths
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        treefile, alnfile = self.pairs[index]
        x, ids = load_alignment(alnfile)
        y = load_distance_matrix(treefile, ids)

        return x, y
