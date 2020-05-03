from data.bigquery_loader import RedditCommentsLoader
from data.federated_datasets import FederatedLanguageDataset, FederatedDataset
from models.lstm_language_model import RNNModel

from typing import List

from torch import nn
from tqdm import tqdm, trange
import yaml

if __name__ == '__main__':

    with open('parameters.yaml') as param_fd:
        parameters = yaml.safe_load(param_fd)

    # configure following
    clients: List[str] = None
    dataset: FederatedDataset = None
    model: nn.Module = None

    if parameters['data']['dataset'] == 'Reddit Comments':
        # download data
        reddit_loader = RedditCommentsLoader(
            table="{}_{}".format(parameters['data']['year'], parameters['data']['month']),
            n_clients=parameters['clients']['n_clients'],
            max_words_per_sample=parameters['clients']['max_words_per_sample'],
            min_words_per_sample=parameters['clients']['min_words_per_sample'],
            max_samples=parameters['clients']['max_samples'],
            min_samples=parameters['clients']['min_samples'],
            train_ratio=parameters['clients']['train_ratio']
        )

        # get client list
        clients = reddit_loader.clients

        # covert to torch data loader
        dataset = FederatedLanguageDataset(
            extraction_directory=reddit_loader.extraction_dir,
            vocab_size=parameters['language_model']['vocab_size'],
            batch_size=parameters['language_model']['batch_size'],
            bptt_len=parameters['language_model']['bptt_len']
        )

        # initialize model
        model = RNNModel(
            rnn_type=parameters['language_model']['rnn_type'],
            vocab_size=parameters['language_model']['vocab_size'],
            embedding_dim=parameters['language_model']['embedding_dim'],
            hidden_dim=parameters['language_model']['hidden_dim'],
            n_layers=parameters['language_model']['n_layers'],
            batch_size=parameters['language_model']['batch_size'],
        )
    else:
        raise KeyError('{} does not exist. Current Supported Datasets: "Redd'.format(parameters['data']['dataset']))

    # start training
    for round in trange(parameters['federated_parameters']['n_rounds'], position=0, desc="Rounds"):
        for client in tqdm(clients, position=1, leave=False, desc="Clients"):
            for epoch in trange(parameters['federated_parameters']['n_epochs'], position=2, leave=False, desc="Epochs"):
                train_iter, test_iter = dataset[client]
                epoch_loss = 0
                for batch in tqdm(train_iter, position=3, leave=False, desc="Batches"):
                    text, target = batch.text, batch.target
                    predictions = model(text,  model.init_hidden())


