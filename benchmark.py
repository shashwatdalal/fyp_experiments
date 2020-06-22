import os

import torch

import yaml
from torch import nn, optim
#from torch.graph_utils.tensorboard import SummaryWriter

from data.bigquery_loader import RedditCommentsLoader
from data.federated_datasets import FederatedLanguageDataset
from models.lstm_language_model import RNNModel
import pandas as pd

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

    save_dir = 'benchmark_models'
    os.makedirs(save_dir, exist_ok=True)

    device = configure_cuda()

    N_EPOCHS = 100

    summary_writer_path = os.path.join('/homes', 'spd16', 'Documents', 'tensorboard')
    #writer = SummaryWriter(summary_writer_path)

    logging_table = pd.DataFrame(columns=['train_loss', 'test_loss', 'acc'], index=pd.Index(clients))

    # train each client model locally
    for client in clients:
        print('training: ', client)

        # initialize model
        model = RNNModel(
            rnn_type=parameters['language_model']['rnn_type'],
            vocab_size=parameters['language_model']['vocab_size'],
            embedding_dim=parameters['language_model']['embedding_dim'],
            hidden_dim=parameters['language_model']['hidden_dim'],
            n_layers=parameters['language_model']['n_layers'],
            batch_size=parameters['language_model']['batch_size'],
        ).to(device)


        for epoch in range(N_EPOCHS):

            optimizer = optim.Adam(model.parameters(), lr=parameters['federated_parameters']['client_lr'])
            loss_fn = nn.NLLLoss()

            train_iter, test_iter = dataset[client]

            train_loss, test_loss, acc = [], [], []

            for i, batch in enumerate(train_iter):
                optimizer.zero_grad()
                text, target = batch.text.to(device), batch.target.to(device)
                predictions, _ = model(text, model.init_hidden())
                loss = loss_fn(predictions, target.view(-1))
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            for batch in test_iter:
                with torch.no_grad():
                    text, target = batch.text.to(device), batch.target.to(device)
                    predictions, _ = model(text, model.init_hidden())
                    test_loss.append(loss_fn(predictions, target.view(-1)).item())
                    acc.append(top3Accuracy(predictions, target))

            current_test_loss = sum(test_loss) / len(test_loss)
            current_train_loss = sum(train_loss) / len(train_loss)
            current_accuracy = sum(acc) / len(acc)

            # writer.add_scalar('{}/train_loss'.format(client), current_train_loss, epoch)
            # writer.add_scalar('{}/test_loss'.format(client), current_test_loss, epoch)
            # writer.add_scalar('{}/test_acc'.format(client), current_accuracy, epoch)


        logging_table.loc[client]['train_loss'] = current_train_loss
        logging_table.loc[client]['test_loss'] = current_test_loss
        logging_table.loc[client]['acc'] = current_accuracy
        # early stopping
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': current_train_loss,
            'test_loss': current_test_loss,
            'test_acc': current_accuracy
        }, os.path.join('benchmark_models', "{}_model.tar".format(client)))

    logging_table.to_csv('benchmark_local_tests.csv')
