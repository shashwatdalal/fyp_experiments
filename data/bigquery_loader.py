import os

from google.cloud import bigquery
import pandas as pd


class BigQueryLoader:

    def __init__(self, extraction_dir, train_ratio, full_file='full.txt', train_file="{}-train.txt",
                 test_file="{}-test.txt",
                 client_field='client', data_field='data'):
        self.data_field = data_field
        self.client_field = client_field
        self.extraction_dir = extraction_dir
        self.train_ratio = train_ratio
        self.full_file = full_file
        self.train_file = train_file
        self.test_file = test_file
        self.load()

    @property
    def query(self):
        raise NotImplementedError

    def load(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'federated-learning-fyp-7a26f33bb71d.json'
        clients = []
        if not os.path.exists(self.extraction_dir):
            # download data if directory does not exist
            os.makedirs(self.extraction_dir, exist_ok=True)
            bq_client = bigquery.Client()
            results_df = bq_client.query(self.query).to_dataframe()
            results_df[self.data_field].to_csv(os.path.join(self.extraction_dir, self.full_file),
                                               header=False, index=False)
            for _, row in results_df.iterrows():
                [client, all_data] = row
                all_data = all_data.split('\n')
                data = []
                # todo dangerous
                tokens_left = self.n_tokens
                for sample in all_data:
                    if len(sample.split(' ')) < tokens_left:
                        data.append(sample)
                        tokens_left -= len(sample.split(' '))
                    else:
                        data.append(' '.join(sample.split(' ')[:tokens_left]))
                        break
                n_train = int(len(data) * self.train_ratio)
                train_data = data[:n_train]
                test_data = data[n_train:]
                pd.Series(train_data).to_csv(os.path.join(self.extraction_dir, self.train_file.format(client)),
                                             header=False, index=False)
                pd.Series(test_data).to_csv(os.path.join(self.extraction_dir, self.test_file.format(client)),
                                            header=False, index=False)
                clients.append(client)
            self.clients = clients
            pd.Series(clients).to_csv(os.path.join(self.extraction_dir, 'clients.txt'),
                                      header=False, index=False)
        else:
            self.clients = list(pd.read_csv(os.path.join(self.extraction_dir, 'clients.txt'),
                                            squeeze=True, header=None))


class RedditCommentsLoader(BigQueryLoader):

    def __init__(self, table, n_clients, n_tokens, train_ratio, root='.data'):
        self.root = root
        self.table = table
        self.n_tokens = n_tokens
        self.n_clients = n_clients
        extraction_dir = os.path.join(self.root, "Reddit-Comments-{}".format(self.table))
        super().__init__(extraction_dir, train_ratio)

    @property
    def query(self):
        return """
        WITH author_tokens as
        (
            SELECT author as {}, STRING_AGG(REPLACE(TRIM(body),\'\\n\',\' \'), \'\\n\') as {}
            FROM reddit_comments.{}
            WHERE author in (
                SELECT DISTINCT author
                FROM reddit_comments.{}
                LIMIT 1000000
            ) 
            GROUP BY author
            HAVING LENGTH(data) - LENGTH(REPLACE(data,\' \',\'\')) >= {}
        )
        select *
        from author_tokens
        limit {} offset 100
        """.format(self.client_field, self.data_field, self.table, self.table, self.n_tokens, self.n_clients)
