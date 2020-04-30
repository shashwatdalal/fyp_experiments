import os
from typing import Tuple, List
from torchtext.data import Iterator, Field, BPTTIterator

from data.language_model_dataset import LocalLanguageModelingDataset


class FederatedLanguageDataset:
    def __getitem__(self, item: str) -> Tuple[Iterator, Iterator]:
        pass

    def get_clients(self) -> List[str]:
        pass

    def build_vocabulary(self):
        raise NotImplementedError


class RedditCommentsDataset(FederatedLanguageDataset, BigQueryLoader):
    def __init__(self, name='reddit-comments', month='2019_10', max_samples=200, min_samples=150, n_clients=1_000,
                 train_ratio=0.8, vocab_size=20_000, batch_size=32, bptt_len=30):
        self.bptt_len = bptt_len
        self.batch_size = batch_size
        self.n_clients = n_clients
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.vocab_size = vocab_size
        self.month = month
        self.extraction_dir = os.path.join('data_extract', '{}-{}'.format(name, month))

        # load data fom BigQuery
        self.clients, self.full_dataset = self.load(self.extraction_dir, train_ratio)

        self.field = Field(lower=True)
        self.build_vocabulary()

    def query(self):
        return """
        SELECT author as client, body
        FROM reddit_comments.{}
        WHERE author IN 
        (
            SELECT author
            FROM reddit_comments.{}
            WHERE author NOT IN ( '[deleted]', 'AutoModerator') 
                and author NOT LIKE '%bot%' 
                and author NOT LIKE '%Bot%' 
                and author NOT LIKE '%BOT%' 
                and author NOT LIKE '%Mod%'
                and body <> ''
            GROUP BY author
            HAVING count(*) < {} and count(*) > {}
            LIMIT {}
        )
        """.format(self.month, self.month, self.max_samples, self.min_samples, self.n_clients)

    def get_clients(self):
        return self.clients

    def build_vocabulary(self):
        all_tokens = LocalLanguageModelingDataset.splits(self.field, root=self.extraction_dir, train=self.full_dataset)
        self.field.build_vocab(all_tokens, max_size=self.vocab_size)

    def __getitem__(self, client: str) -> Tuple[Iterator, Iterator]:
        train, test = LocalLanguageModelingDataset.splits(self.field, root=self.extraction_dir,
                                                          train="{}-train.txt".format(client),
                                                          test="{}-test.txt".format(client))
        return BPTTIterator.splits((train, test), batch_size=self.batch_size, bptt_len=self.bptt_len)
