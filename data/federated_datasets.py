from typing import Tuple

from torchtext.data import Iterator, Field, BPTTIterator
from torchtext.datasets import LanguageModelingDataset


class LocalLanguageModelingDataset(LanguageModelingDataset):
    name = ''
    dirname = ''
    urls = []

    @classmethod
    def splits(cls, text_field, root='.data', train=None, test=None, **kwargs):
        return super(LocalLanguageModelingDataset, cls).splits(
            root=root, train=train, test=test,
            text_field=text_field, **kwargs)


class FederatedDataset:
    def __getitem__(self, client: str) -> Tuple[Iterator, Iterator]:
        """
            client to train/test data loader
        """
        raise NotImplementedError


class FederatedLanguageDataset(FederatedDataset):
    def __init__(self, extraction_directory, vocab_size, batch_size, bptt_len):
        self.extraction_directory = extraction_directory
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.bptt_len = bptt_len

        self.field = Field(lower=True)
        self.build_vocab()

    def build_vocab(self):
        all_tokens = LocalLanguageModelingDataset.splits(self.field, root=self.extraction_directory, train='full.txt')
        self.field.build_vocab(all_tokens[0], max_size=self.vocab_size)

    def __getitem__(self, client: str) -> Tuple[Iterator, Iterator]:
        train, test = LocalLanguageModelingDataset.splits(self.field, root=self.extraction_directory,
                                                          train="{}-train.txt".format(client),
                                                          test="{}-test.txt".format(client))
        return BPTTIterator.splits((train, test), batch_size=self.batch_size, bptt_len=self.bptt_len)
