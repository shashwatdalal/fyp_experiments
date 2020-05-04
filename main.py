from data.bigquery_loader import RedditCommentsLoader
from data.federated_datasets import FederatedLanguageDataset, FederatedDataset
from models.lstm_language_model import RNNModel

import copy
from typing import List

import torch
from torch import nn, optim
from tqdm import tqdm, trange
import yaml

TQDM = False
GPU = True


def configure_cuda():
    device_idx = 0
    if GPU:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # We set a random seed to ensure that your results are reproducible.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    return device


if __name__ == '__main__':

    with open('parameters.yaml') as param_fd:
        parameters = yaml.safe_load(param_fd)

    # configure following
    clients: List[str] = None
    dataset: FederatedDataset = None
    server_model: nn.Module = None

    # configure GPU Settings
    device = configure_cuda()

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
        server_model = RNNModel(
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
    for round in trange(parameters['federated_parameters']['n_rounds'], position=0, desc="Rounds", disable=not TQDM):

        client_updates = {name: torch.zeros(len(clients), *param.shape) for name, param in
                          server_model.named_parameters()}

        # perform training
        for i, client in enumerate(tqdm(clients, position=1, leave=False, desc="Clients", disable=not TQDM)):

            # 'send' client server model
            client_model = copy.deepcopy(server_model).to(device)

            # initialize optimizer and loss function
            client_optimizer = optim.Adam(client_model.parameters(), lr=0.01)
            loss_fn = nn.CrossEntropyLoss()

            # perform local update
            train_loss, test_loss = [], []
            for epoch in trange(parameters['federated_parameters']['n_epochs'],
                                position=2, leave=False, desc="Epochs", disable=not TQDM):
                train_iter, test_iter = dataset[client]
                for batch in tqdm(train_iter, position=3, leave=False, desc="Train Batch", disable=not TQDM):
                    client_optimizer.zero_grad()
                    text, target = batch.text.to(device), batch.target.to(device)
                    predictions, _ = client_model(text, client_model.init_hidden())
                    loss = loss_fn(predictions, target.view(-1))
                    train_loss.append(loss.item())
                    loss.backward()
                    client_optimizer.step()
            print(round, client, sum(train_loss) / len(train_loss))

                # # calculate test loss
                # for batch in tqdm(test_iter, position=3, leave=False, desc="Test Batch", disable=not TQDM):
                #     with torch.no_grad():
                #         text, target = batch.text, batch.target
                #         predictions, _ = client_model(text, client_model.init_hidden())
                #         test_loss.append(loss_fn(predictions, target.view(-1)))

                # todo log test/train loss

            # 'send' server update
            for (name, client_param), server_param in zip(client_model.named_parameters(), server_model.parameters()):
                client_updates[name][i] = client_param.detach().cpu() - server_param.detach()
            # print('update', client, size_of_update)

        # aggregate model
        with torch.no_grad():
            for name, server_param in server_model.named_parameters():
                server_param.data = server_param.data + torch.mean(client_updates[name], dim=0)
