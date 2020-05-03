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
                [client, data] = row
                data = data.split('\n')
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
            self.clients = list(
                pd.read_csv(os.path.join(self.extraction_dir, 'clients.txt'),
                            squeeze=True, header=None))


class RedditCommentsLoader(BigQueryLoader):

    def __init__(self, table, n_clients, max_words_per_sample, min_words_per_sample,
                 max_samples, min_samples, train_ratio, root='.data'):
        self.root = root
        self.table = table
        self.n_clients = n_clients
        self.max_words_per_sample = max_words_per_sample
        self.min_words_per_sample = min_words_per_sample
        self.max_samples = max_samples
        self.min_samples = min_samples
        extraction_dir = os.path.join(self.root, "Reddit-Comments-{}".format(self.table))
        super().__init__(extraction_dir, train_ratio)

    @property
    def query(self):
        return """
            SELECT author as {}, STRING_AGG(REPLACE(TRIM(body),\'\\n\',\' \'), \'\\n\') as {}
            FROM reddit_comments.{} 
            WHERE 
                LENGTH(TRIM(body)) - LENGTH(REPLACE(TRIM(body),' ','')) < {}
                and 
                LENGTH(TRIM(body)) - LENGTH(REPLACE(TRIM(body),' ','')) > {} 
            GROUP BY author 
            HAVING 
                count(*) < {} 
                and 
                count(*) > {}
            LIMIT {}
        """.format(self.client_field, self.data_field, self.table,
                   self.max_words_per_sample, self.min_words_per_sample,
                   self.max_samples, self.min_samples, self.n_clients)
