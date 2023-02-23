"""
Este código tem por objetivo criar um modelo de autoencoder para a tarefa de
diminuir a dimensionalidade dos vetores de embeddings gerados pelo BERT dos
textos de reviews de restaurantes.
"""

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, input_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        train_loss += loss.item() * data.size(0)
        optimizer.step()
    return train_loss / len(train_loader.dataset)