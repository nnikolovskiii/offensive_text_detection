import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, output_dim, n_layers, dropout):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)



    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        last_hidden_layer = output[:, -1, :]

        hidden = self.dropout(last_hidden_layer)

        output = self.fc_out(hidden)

        return output