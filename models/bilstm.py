import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM model for sequence classification/labeling.
    Built using PyTorch nn.LSTM.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Forward pass for BiLSTM.
        """
        pass
