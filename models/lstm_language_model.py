import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, n_layers, batch_size, dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_dim != embedding_dim:
                raise ValueError('When using the tied flag, hidden_dim must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, self.batch_size, self.hidden_dim),
                    weight.new_zeros(self.n_layers, self.batch_size, self.hidden_dim))
        else:
            return weight.new_zeros(self.n_layers, self.batch_size, self.hidden_dim)
