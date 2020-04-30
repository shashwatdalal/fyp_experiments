import os

from google.cloud import bigquery
import pandas as pd


class BigQueryLoader:

    def query(self):
        raise NotImplementedError

    def load(self, extraction_dir, train_ratio, full_file='full.txt', train_file="{}-train.txt", test_file="{}-test.txt"):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'federated-learning-fyp-7a26f33bb71d.json'
        clients = []
        if not os.path.exists(extraction_dir):
            os.makedirs(extraction_dir, exist_ok=True)
            client = bigquery.Client()
            results_df: pd.DataFrame = client.query(self.query()).to_dataframe()
            results_df['body'].to_csv(os.path.join(extraction_dir, full_file), header=False, index=False)
            for client, examples in results_df.groupby('client'):
                n_train = int(len(examples) * train_ratio)
                train_df = examples[:n_train]
                test_df = examples[n_train:]
                train_df['body'].to_csv(os.path.join(extraction_dir, train_file.format(client)), header=False,
                                        index=False)
                test_df['body'].to_csv(os.path.join(extraction_dir, test_file.format(client)), header=False,
                                       index=False)
                clients.append(client)
            print(clients)
            pd.Series(clients).to_csv(os.path.join(extraction_dir, 'clients.txt'), header=False, index=False)
        else:
            clients = list(pd.read_csv(os.path.join(extraction_dir, 'clients.txt'), squeeze=True))
        return clients, full_file
