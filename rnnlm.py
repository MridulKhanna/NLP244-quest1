import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting the value of constants
type_of_rnn = "lstm"
bidirectional_flag = True


temp_batch_size = 0
if bidirectional_flag == True:
    num_directions = 2
else:
    num_directions = 1

class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        in_embedding_dim,
        n_hidden,
        n_layers,
        dropout=0.5,
        rnn_type=type_of_rnn,  # can be elman, lstm, gru
    ):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        if rnn_type == "elman":
            self.rnn = nn.RNN(
                in_embedding_dim,
                n_hidden,
                n_layers,
                nonlinearity="tanh",
                dropout=dropout,
                bidirectional = bidirectional_flag
            )
        elif rnn_type == "lstm":
            # TODO: implement lstm and gru
            # self.rnn = ...
            self.rnn = nn.LSTM(in_embedding_dim, n_hidden, n_layers, dropout=dropout, bidirectional = bidirectional_flag)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(in_embedding_dim, n_hidden, n_layers, dropout=dropout, bidirectional = bidirectional_flag)
        
        self.in_embedder = nn.Embedding(vocab_size, in_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.Linear(n_hidden * num_directions, vocab_size)
        self.init_weights()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.in_embedding_dim = in_embedding_dim

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.in_embedder.weight, -initrange, initrange)
        nn.init.zeros_(self.pooling.bias)
        nn.init.uniform_(self.pooling.weight, -initrange, initrange)

    def forward(self, input, hidden):
        global temp_batch_size
        emb = self.dropout(self.in_embedder(input))
        if self.rnn_type == "lstm":
            h_0 = torch.zeros(num_directions * self.n_layers, temp_batch_size, self.n_hidden).cuda()
            c_0 = torch.zeros(num_directions * self.n_layers, temp_batch_size, self.n_hidden).cuda()
            output, hidden = self.rnn(emb, (h_0, c_0))
        else:
            output, hidden = self.rnn(emb, hidden)

        output = self.dropout(output)
        pooled = self.pooling(output)
        pooled = pooled.view(-1, self.vocab_size)
        return F.log_softmax(pooled, dim=1), hidden

    def init_hidden(self, batch_size):
        global temp_batch_size
        temp_batch_size = batch_size
        weight = next(self.parameters())
        if self.rnn_type == "gru" or self.rnn_type == "elman":
            return weight.new_zeros(self.n_layers * num_directions, batch_size, self.n_hidden)
        else:
            return weight.new_zeros(self.n_layers, batch_size, self.n_hidden)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = torch.load(f)
            model.rnn.flatten_parameters()
            return model
