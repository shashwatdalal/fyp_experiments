import io
import random
from typing import Tuple

from torchtext import data
from torchtext.data import Iterator, Field, BPTTIterator


class LocalLanguageModelingDataset(data.Dataset):
    name = ''
    dirname = ''
    urls = []

    def __init__(self, path, text_field, newline_eos=True,
                 encoding='utf-8', shuffle=False, **kwargs):
        """Create a LanguageModelingDataset given a path and a field.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.f
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field)]
        text = []
        with io.open(path, encoding=encoding) as f:
            for i, line in enumerate(f):
                line_ = []
                line_ += text_field.preprocess(line)
                if newline_eos:
                    line_.append(u'<eos>')
                text.append(line_)
        if shuffle:
            random.shuffle(text)
            print(text[:5])
        text = [token for line in text for token in line]
        examples = [data.Example.fromlist([text], fields)]
        super(LocalLanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, root='.data', train=None, test=None, shuffle=False, **kwargs):
        return super(LocalLanguageModelingDataset, cls).splits(
            root=root, train=train, test=test,
            text_field=text_field, shuffle=shuffle, **kwargs)


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
        self.field = Field(lower=True, tokenize='basic_english', stop_words={',', '.', '!', '?', '&gt', '\'', '(', ')'})
        self.build_vocab()

    def build_vocab(self):
        all_tokens = LocalLanguageModelingDataset.splits(self.field, root=self.extraction_directory, train='full.txt')
        self.field.build_vocab(all_tokens[0], max_size=self.vocab_size)

    def __getitem__(self, client: str) -> Tuple[Iterator, Iterator]:
        train, test = LocalLanguageModelingDataset.splits(self.field, root=self.extraction_directory,
                                                          train="{}-train.txt".format(client),
                                                          test="{}-test.txt".format(client), shuffle=True)
        return BPTTIterator.splits((train, test), batch_size=self.batch_size, bptt_len=self.bptt_len)
