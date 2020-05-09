import os

from data.bigquery_loader import RedditCommentsLoader
from data.federated_datasets import FederatedLanguageDataset, FederatedDataset
from models.lstm_language_model import RNNModel

import copy
import time
from itertools import product
from random import sample
from typing import List

import pandas as pd
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
    print('device:', device)

    # We set a random seed to ensure that your results are reproducible.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    return device


def top3Accuracy(predictions, target):
    in_top3 = (torch.topk(predictions, k=3, dim=1).indices == target.view(-1)[..., None]).any(-1)
    return torch.sum(in_top3).item() / len(in_top3)


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
            n_tokens=parameters['clients']['n_tokens'],
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

    # logging table
    model_param_names, _ = zip(*server_model.named_parameters())
    metrics = list(model_param_names) + \
              ['pre_test_acc', 'post_test_acc', 'train_loss', 'pre_test_loss', 'post_test_loss']
    clients_p_round = parameters['federated_parameters']['clients_p_round']
    column_names = product(metrics, clients)
    logging_table = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(column_names),
        index=pd.Index(range(parameters['federated_parameters']['n_rounds']))
    )

    # start training
    for round in trange(parameters['federated_parameters']['n_rounds'], position=0, desc="Rounds", disable=not TQDM):

        client_updates = {name: torch.zeros(clients_p_round, *param.shape) for name, param in
                          server_model.named_parameters()}

        # perform training
        for i, client in enumerate(
                tqdm(sample(clients, clients_p_round), position=1, leave=False, desc="Clients", disable=not TQDM)):
            # 'send' client server model
            client_model = copy.deepcopy(server_model).to(device)

            # initialize optimizer and loss function
            client_optimizer = optim.Adam(client_model.parameters(), lr=parameters['federated_parameters']['client_lr'])
            loss_fn = nn.NLLLoss()

            train_loss, pre_test_loss, pre_test_accuracy, post_test_loss, post_test_accuracy = [], [], [], [], []

            test_iter, train_iter = dataset[client]

            # calculate pre test loss
            for batch in tqdm(test_iter, position=3, leave=False, desc="Test Batch", disable=not TQDM):
                train_iter, test_iter = dataset[client]
                with torch.no_grad():
                    text, target = batch.text.to(device), batch.target.to(device)
                    predictions, _ = client_model(text, client_model.init_hidden())
                    pre_test_loss.append(loss_fn(predictions, target.view(-1)).item())
                    pre_test_accuracy.append(top3Accuracy(predictions, target))
            logging_table.loc[round][('pre_test_loss', client)] = sum(pre_test_loss) / len(pre_test_loss)
            logging_table.loc[round][('pre_test_acc', client)] = sum(pre_test_accuracy) / len(pre_test_accuracy)

            # train
            for epoch in trange(parameters['federated_parameters']['n_epochs'],
                                position=2, leave=False, desc="Epochs", disable=not TQDM):
                for batch in tqdm(train_iter, position=3, leave=False, desc="Train Batch", disable=not TQDM):
                    client_optimizer.zero_grad()
                    text, target = batch.text.to(device), batch.target.to(device)
                    predictions, _ = client_model(text, client_model.init_hidden())
                    loss = loss_fn(predictions, target.view(-1))
                    train_loss.append(loss.item())
                    loss.backward()
                    client_optimizer.step()
            logging_table.loc[round][('train_loss', client)] = sum(train_loss) / len(train_loss)

            # calculate post test loss
            for batch in tqdm(test_iter, position=3, leave=False, desc="Test Batch", disable=not TQDM):
                with torch.no_grad():
                    text, target = batch.text.to(device), batch.target.to(device)
                    predictions, _ = client_model(text, client_model.init_hidden())
                    post_test_loss.append(loss_fn(predictions, target.view(-1)).item())
                    post_test_accuracy.append(top3Accuracy(predictions, target))

            # 'send' server update
            for (name, client_param), server_param in zip(client_model.named_parameters(), server_model.parameters()):
                client_updates[name][i] = client_param.detach().cpu() - server_param.detach()
                logging_table.loc[round][(name, client)] = torch.norm(client_updates[name][i], 2).item()
            logging_table.loc[round][('post_test_loss', client)] = sum(post_test_loss) / len(post_test_loss)
            logging_table.loc[round][('post_test_acc', client)] = sum(post_test_accuracy) / len(post_test_accuracy)

        identifier = 'clients_{}_q_{}_epoch_{}_lr_{}'.format(
            parameters['clients']['n_clients'],
            parameters['federated_parameters']['clients_p_round'],
            parameters['federated_parameters']['n_epochs'],
            parameters['federated_parameters']['client_lr'])

        # aggregate model
        sim_dir = os.path.join("SIMILARITY_MEASURE_{}".format(identifier), "round_{}".format(round))
        os.makedirs(sim_dir, exist_ok=True)
        start = time.process_time()
        with torch.no_grad():
            for name, server_param in server_model.named_parameters():
                server_param.data = server_param.data + torch.mean(client_updates[name], dim=0)
                n_clients = client_updates[name].shape[0]
                vectorized_update = client_updates[name].view(n_clients, -1).to(device)
                similarity_measure = vectorized_update @ vectorized_update.T
                torch.save(similarity_measure, os.path.join(sim_dir, "{}.pt".format(name)))

        logging_table.to_csv('METRICS_clients_{}_q_{}_epoch_{}_lr_{}.csv'.format(
            parameters['clients']['n_clients'],
            parameters['federated_parameters']['clients_p_round'],
            parameters['federated_parameters']['n_epochs'],
            parameters['federated_parameters']['client_lr']
        ))
