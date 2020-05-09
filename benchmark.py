import glob
import torch

import yaml
from torch import nn, optim
from torchtext.data import Field, BPTTIterator
import pandas as pd

from data.federated_datasets import LocalLanguageModelingDataset
from models.lstm_language_model import RNNModel

EPOCHS = 200
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

    model: nn.Module = RNNModel(
        rnn_type=parameters['language_model']['rnn_type'],
        vocab_size=parameters['language_model']['vocab_size'],
        embedding_dim=parameters['language_model']['embedding_dim'],
        hidden_dim=parameters['language_model']['hidden_dim'],
        n_layers=parameters['language_model']['n_layers'],
        batch_size=parameters['language_model']['batch_size'],
    )

    device = configure_cuda()

    train_files = glob.glob('.data/Reddit-Comments-2019_10/*-train.txt')

    with open('.data/train.txt', 'w') as outfile:
        for f in train_files:
            with open(f, "r") as infile:
                outfile.write(infile.read())

    test_files = glob.glob('.data/Reddit-Comments-2019_10/*-test.txt')

    with open('.data/test.txt', 'w') as outfile:
        for f in train_files:
            with open(f, "r") as infile:
                outfile.write(infile.read())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    field = Field(lower=True, tokenize='basic_english')
    train, test = LocalLanguageModelingDataset.splits(field, root='.data', train='train.txt', test='test.txt')
    field.build_vocab(train, test, max_size=parameters['language_model']['vocab_size'])

    logging_table = pd.DataFrame(columns=['test_acc, train_loss, test_loss'], index=range(EPOCHS))

    for e in range(EPOCHS):
        print(e)
        train_loss, test_loss, test_acc = [], [], []
        test_iter, train_iter = BPTTIterator.splits((train, test), batch_size=32, bptt_len=20)
        for batch in train_iter:
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
                test_acc.append(top3Accuracy(predictions, target))

        logging_table.loc[e]['train_loss'] = sum(train_loss) / len(train_loss)
        logging_table.loc[e]['test_loss'] = sum(test_loss) / len(test_loss)
        logging_table.loc[e]['test_acc'] = sum(test_acc) / len(test_acc)
        logging_table.to_csv('benchmark.csv')
