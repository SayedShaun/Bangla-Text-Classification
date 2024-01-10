from torch import nn


class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, layer_size, nums_class):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, layer_size)
        self.fc = nn.Linear(hidden_size, nums_class)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        linear = self.fc(hidden)
        return linear
