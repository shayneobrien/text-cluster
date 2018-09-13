import torch
import torch.nn as nn
import numpy as np
from datasketch import MinHashLSH
from .modules import LazyVectors

VECTORS = LazyVectors(cache='/Users/sob/github/.vector_cache/')

def token_to_id(token):
    """ Lookup word ID for a token """
    return VECTORS.stoi(token)

def sent_to_tensor(sent):
    """ Convert a sentence to a lookup ID tensor """
    idx_tensor = torch.tensor([token_to_id(t) for t in sent.split()])
    if idx_tensor.shape[0] == 0: # Empty, edge case; return UNK
        return torch.tensor([VECTORS.unk_idx])
    else:
        return idx_tensor


class CBoW(nn.Module):
    """ Continuous Bag of Words (sentence representation is average of
    word vectors)
    """
    def __init__(self):
        super().__init__()

        weights = VECTORS.weights()
        self.embeddings = nn.Embedding(weights.shape[0],
                                       weights.shape[1],
                                       padding_idx=0)
        self.embeddings.weight.data.copy_(weights)

    def forward(self, batch):
        """ Accepts list of strings, embeds them using GLoVe,
        averages the word vectors (including UNK tokens),
        returns as numpy array for input into clustering classes """
        tensors = [sent_to_tensor(sent) for sent in batch]
        embedded = [self.embeddings(t) for t in tensors]
        return np.vstack([t.mean(dim=0).detach().numpy() for t in embedded])
