import torch
import torch.nn as nn
from torchtext.vocab import Vectors

import pandas as pd
from cached_property import cached_property
from datasketch import MinHash, MinHashLSH
from .utils import clean_string, read_file


class StreamData:
    """ Stream in data sequentially. Uses pandas dataframes for intra-batch
    deduping and optional lsh_hash for historical deduping (sublinear complexity)
    """
    def __init__(self, filename, chunk=250, min_len=25, clean_fnc=clean_string,
                        lsh_hash=True, use_column=None):

        self.__dict__.update(locals())
        self.generator = pd.read_csv(filename, chunksize=chunk)

        self.n_processed = 0
        if self.lsh_hash == True:
            self.lsh_hash = MinHashLSH(threshold=0.995, num_perm=128)

    def __call__(self):
        """ Get a batch from the generator """
        return self._process(self.stream())

    def stream(self):
        """ Iterate generator """
        return next(self.generator)

    def _init_data(self, num_chunks):
        """ Generate a bunch of data to serve as initialization """
        return pd.concat([self.__call__() for _ in range(num_chunks)])

    def _process(self, batch):
        """ If use_column is specified, use that to make a new column of
        processed text data, remove rows where processed is less than min_len.
        From the resulting dataframe, remove duplicates. """
        if self.use_column is not None:
            batch = batch.assign(processed=self._clean(batch[self.use_column]))
            batch = batch[[len(s.split()) > self.min_len for s in batch.processed]]
        deduped = self._dedupe(batch)
        self.n_processed += len(deduped)
        return deduped

    def _clean(self, batch):
        """ Clean data using some function """
        if self.clean_fnc is not None:
            return [self.clean_fnc(sent) for sent in batch]
        return batch

    def _dedupe(self, dataframe):
        """ Delete duplicates of a dataframe. If use_column is specified,
        operate on the processed text. After deduping within dataframe,
        if lsh_hash is enabled check to make sure rows have also not
        already been seen before. """
        if self.use_column is not None:
            deduped = dataframe.drop_duplicates(subset=['processed'])
            if type(self.lsh_hash) == MinHashLSH:
                deduped = self._hash(dataframe)
        else:
            deduped = dataframe.drop_duplicates()

        deduped.index = range(self.n_processed, self.n_processed+len(deduped))

        return deduped

    def _hash(self, dataframe):
        """ Process dataframe to delete duplicates based on Jaccard similarity
        then update hash """
        # Convert current batch to hash table
        hash_batch = self._batch_to_hash(dataframe.processed, dataframe.index)

        # Greedy, locally sensitive query to see if its a duplicate
        kept_hashes, kept_idx = self._query_hash(hash_batch)

        # Keep only non-duplicates
        dataframe = dataframe[kept_idx]

        # Get new indexes for updates to keep things consistent in lsh dict
        indexes = range(self.n_processed, self.n_processed+len(dataframe))

        # Realign the kept hash update keys
        updates = [(i, h[1]) for i, h in zip(indexes, kept_hashes)]

        # Update the hash table
        self._update_hash(updates)

        return dataframe

    def _update_hash(self, hash_batch):
        """ After processing a batch, update lsh_hash with new entries """
        with self.lsh_hash.insertion_session() as session:
            for idx, hasher in hash_batch:
                session.insert(idx, hasher)

    def _query_hash(self, hash_batch):
        """ Query lsh_hash and ignore entries that have already been seen """
        keep_hashes, keep_idx = [], []
        for hasher in hash_batch:
            if not self.lsh_hash.query(hasher[1]):
                keep_hashes.append(hasher)
                keep_idx.append(True)
            else:
                keep_idx.append(False)

        return keep_hashes, keep_idx

    def _batch_to_hash(self, batch, indexes):
        """ Convert a list of strings to a list of
        tuples (index, hash object)"""
        return [(idx, self._str_to_hash(string))
                for idx, string in zip(indexes, batch)]

    def _str_to_hash(self, string):
        """ Convert string to locality sensitive min-hash """
        data = set(string.split())
        hasher = MinHash(num_perm=128)
        for d in data:
            hasher.update(d.encode('utf-8'))
        return hasher


class LazyVectors:
    """Load only those vectors from GloVE that are in the vocab."""

    unk_idx = 1

    def __init__(self, name='glove.840B.300d.txt',
                         cache='../../github/.vector_cache/',
                         vocab_file='../data/vocabulary.txt',
                         skim=None,
                         vocab=None):
        """  Requires the glove vectors to be in a folder named .vector_cache

        Setup:
            >> cd ~/where_you_want_to_save_glove_vectors
            >> mkdir .vector_cache
            >> mv ~/where_glove_vectors_are_stored/glove.840B.300d.txt
                ~/where_you_want_to_save_glove_vectors/.vector_cache/glove.840B.300d.txt

        Initialization (first init will be slow):
            >> VECTORS = LazyVectors(cache='~/where_you_saved_to/.vector_cache/',
                                     vocab_file='../path/vocabulary.txt',
                                     skim=None)

        Usage:
            >> weights = VECTORS.weights()
            >> embeddings = torch.nn.Embedding(weights.shape[0],
                                              weights.shape[1],
                                              padding_idx=0)
            >> embeddings.weight.data.copy_(weights)
            >> embeddings(sent_to_tensor('kids love unknown_word food'))

        You can access these moved vectors from any repository
        """
        self.__dict__.update(locals())
        if not self.vocab:
            self.set_vocab()

    @cached_property
    def loader(self):
        return Vectors(self.name, cache=self.cache)

    def set_vocab(self):
        """Set corpus vocab """
        # Intersect with model vocab.
        try:
            cached_vocab = self.get_vocab(self.vocab_file)
        except FileNotFoundError:
            raise AttributeError('Must provide list/set of vocab or a \
                                 valid filename to read from.')

        # Intersect
        self.vocab = [v for v in cached_vocab
                      if v in self.loader.stoi][:self.skim]

        self.set_dicts()

    def get_vocab(self, filename):
        """ Read in vocabulary (top 30K words, covers ~93.5% of all tokens) """
        return read_file(filename)

    def set_dicts(self):
        """ _stoi: map string > index
            _itos: map index > string
        """
        self._stoi = {s: i for i, s in enumerate(self.vocab)}
        self._itos = {i: s for s, i in self._stoi.items()}

    def weights(self):
        """Build weights tensor for embedding layer """
        # Select vectors for vocab words.
        weights = torch.stack([
            self.loader.vectors[self.loader.stoi[s]]
            for s in self.vocab
        ])

        # Padding + UNK zeros rows.
        return torch.cat([
            torch.zeros((2, self.loader.dim)),
            weights,
        ])

    def stoi(self, s):
        """ String to index (s to i) for embedding lookup """
        idx = self._stoi.get(s)
        return idx + 2 if idx else self.unk_idx

    def itos(self, i):
        """ Index to string (i to s) for embedding lookup """
        token = self._itos.get(i)
        return token if token else 'UNK'
