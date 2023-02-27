import torch


class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            input_size,
            conv_layers,
            conv_sizes,
            hidden_layers_encoder,
            hidden_layers_decoder,
            latent_dim,
            dropout
    ):
        super(Autoencoder, self).__init__()

        conv_layers = [1] + conv_layers
        self.conv_net = []
        for i in range(0, len(conv_layers)-1):
            self.conv_net += [
                torch.nn.Conv1d(conv_layers[i], conv_layers[i+1], kernel_size=conv_sizes[i], padding='same'),
                torch.nn.BatchNorm1d(conv_layers[i+1]),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU()
            ]

        self.encoder = []
        self.decoder = []

        hidden_layers_encoder = [conv_layers[-1] * input_size] + hidden_layers_encoder + [latent_dim]
        hidden_layers_decoder = [latent_dim] + hidden_layers_decoder + [input_size]

        for i in range(0, len(hidden_layers_encoder)-1):
            self.encoder += [
                torch.nn.Linear(hidden_layers_encoder[i], hidden_layers_encoder[i+1]),
                torch.nn.ReLU()
            ]

        for i in range(0, len(hidden_layers_decoder)-1):
            self.decoder += [
                torch.nn.Linear(hidden_layers_decoder[i], hidden_layers_decoder[i+1]),
                torch.nn.ReLU()
            ]

        self.conv_net = torch.nn.Sequential(*self.conv_net)
        self.encoder = torch.nn.Sequential(*self.encoder)
        self.decoder = torch.nn.Sequential(*self.decoder)

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        h = self.conv_net(x)
        # Flatten the output for dense layer entry
        h = h.view(h.shape[0], -1)
        encoded = self.encoder(h)
        decoded = self.decoder(encoded)
        return decoded


def train(model, optimizer, data_train_loader, n_epoch):
    for epoch in range(n_epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_train_loader):
            optimizer.zero_grad()

            reconstructed = model(data)
            loss = model.loss_fn(reconstructed, data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('[*] Epoch: {} - Average loss: {:.4f}'.format(epoch+1, train_loss / len(data_train_loader.dataset)))

    return model


####################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from problem import get_train_data, get_test_data

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_cgm_data():
    data = pd.read_csv('external_data.csv')
    data.set_index('patient_id', inplace=True)
    return data


def get_patient_cgm_data(patient_id):
    cgm_data = get_cgm_data()
    patient_cgm = cgm_data.loc[patient_id]
    patient_cgm.dropna(inplace=True)
    return patient_cgm


class CGM_dataset(Dataset):

    def __init__(self, cgm_data):
        self.x = torch.tensor(cgm_data.to_numpy(), dtype=torch.float32).reshape(-1, 1, len(cgm_data)).permute(2, 1, 0)
        self.y = torch.tensor(cgm_data.to_numpy(), dtype=torch.float32).reshape(-1, 1, len(cgm_data)).permute(2, 1, 0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


all_cgm_data = get_cgm_data()

all_cgm_data = all_cgm_data.dropna(axis=0)

X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

train_cgm_data = CGM_dataset(all_cgm_data.loc[X_train.index.intersection(all_cgm_data.index)])
test_cgm_data = CGM_dataset(all_cgm_data.loc[X_test.index.intersection(all_cgm_data.index)])

train_loader = DataLoader(train_cgm_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_cgm_data, batch_size=32, shuffle=False)

model = Autoencoder(
    input_size=576,
    conv_layers=[300, 200, 100],
    conv_sizes=[9, 33, 77],
    hidden_layers_encoder=[288, 128],
    hidden_layers_decoder=[128, 288],
    latent_dim=64,
    dropout=0.3
)

optimizer = torch.optim.Adam(model.parameters(), lr=10e-3, weight_decay=10e-4)

train(model, optimizer, train_loader, n_epoch=100)

reconstructed_cgm_data = []
max_plot = 30
for batch_index, (cgm_entry_batch, _) in enumerate(test_loader):
    n_plots = 1

    for cgm_entry in cgm_entry_batch:
        reconstructed_cgm_data = model(cgm_entry.view(1, 1, -1))
        reconstructed_cgm_data = reconstructed_cgm_data.detach().numpy().squeeze()
        cgm_entry = cgm_entry.detach().numpy().squeeze()

        fig, axs = plt.subplots()
        axs.plot(cgm_entry, label="Original CGM data")
        axs.plot(reconstructed_cgm_data, label="Reconstructed CGM data", color="r")
        axs.legend()
        n_plots += 1
        if n_plots >= max_plot:
            break
    if n_plots >= max_plot:
        break

plt.show()
