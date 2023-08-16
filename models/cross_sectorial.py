from typing import Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from pytorch_ranger import Ranger
from ranger21 import Ranger21
from torch import nn, optim
import wandb

##############################################################################################
##############################################################################################
######################################## Vanilla LSTM ########################################
##############################################################################################
##############################################################################################


class LSTM(pl.LightningModule):
    def __init__(self, 
                 n_companies,
                 n_features,
                 lookback,
                 epochs,
                 batches_p_epoch,
                 lstm_layers=2, 
                 lstm_nodes=64, 
                 fc_layers=1,
                 fc_nodes=64,
                 dropout=0, 
                 bidirectional=True,
                 lr=1e-3, 
                 optimizer=Ranger21,
                 activation=nn.Mish(True),
                 loss_fn=nn.MSELoss()):
        
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["activation", "loss_fn"])

        self.n_companies = n_companies
        self.n_features = n_features
        self.lookback = lookback
        self.epochs = epochs
        self.batches_p_epoch = batches_p_epoch
        self.lstm_layers = lstm_layers
        self.lstm_nodes = lstm_nodes
        self.fc_layers = fc_layers
        self.fc_nodes = fc_nodes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.optimizer = optimizer
        self.activation = activation
        self.loss_fn = loss_fn

        self.lstm = nn.LSTM(input_size=self.n_features, 
                            hidden_size=self.lstm_nodes, 
                            num_layers=self.lstm_layers, 
                            batch_first=True, 
                            dropout=self.dropout, 
                            bidirectional=self.bidirectional)

        if self.fc_layers == 1:
            self.fc = nn.Linear(self.lstm_nodes*(1+self.bidirectional), self.n_companies)
        else:
            lin_layers = []
            for i in range(self.fc_layers):

                if i == 0:
                    lin_layers.append(nn.Linear(self.lstm_nodes*(1+self.bidirectional), self.fc_nodes))
                    lin_layers.append(self.activation)
                    lin_layers.append(nn.Dropout(self.dropout))
                else:
                    lin_layers.append(nn.Linear(self.fc_nodes, self.fc_nodes))
                    lin_layers.append(self.activation)
                    lin_layers.append(nn.Dropout(self.dropout))

            self.fc = nn.Sequential(*lin_layers, nn.Linear(self.fc_nodes, self.n_companies))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        # Ranger requires manual backward pass to get along with lightning (also, Ranger doesnt work with torch compile)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X).squeeze(1)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        if self.optimizer == Ranger21:
            optimizer = self.optimizer(self.parameters(), lr=self.lr, num_epochs=self.epochs, num_batches_per_epoch=self.batches_p_epoch)
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
    

##############################################################################################
##############################################################################################
###################################### 1D CNN LSTM Model #####################################
##############################################################################################
##############################################################################################


class CNN_1D_LSTM(pl.LightningModule):
    def __init__(self, 
                 n_companies,
                 n_features,
                 lookback,
                 epochs,
                 batches_p_epoch,
                 lstm_layers=2, 
                 lstm_nodes=64, 
                 fc_layers=1,
                 fc_nodes=64,
                 dropout=0, 
                 bidirectional=True,
                 lr=1e-3, 
                 optimizer=Ranger21,
                 activation=nn.Mish(True),
                 loss_fn=nn.MSELoss()):
        
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["activation", "loss_fn"])

        self.n_companies = n_companies
        self.n_features = n_features
        self.lookback = lookback
        self.epochs = epochs
        self.batches_p_epoch = batches_p_epoch
        self.lstm_layers = lstm_layers
        self.lstm_nodes = lstm_nodes
        self.fc_layers = fc_layers
        self.fc_nodes = fc_nodes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.optimizer = optimizer
        self.activation = activation
        self.loss_fn = loss_fn

        # Feature selection and dimensionality reduction using 1D convolutions
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=64, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=self.last_conv_size, kernel_size=1),
            nn.ReLU(True),
        )


        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0,
            bidirectional= self.bidirectional
        )

        # Fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.output_size),
        )

    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        out = out.permute(0, 2, 1)
        # # # # LSTM
        out, _ = self.lstm(out)
        out = self.fc2(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        # Ranger requires manual backward pass to get along with lightning (also, Ranger doesnt work with torch compile)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        if self.optimizer == Ranger21:
            optimizer = self.optimizer(self.parameters(), lr=self.lr, num_epochs=self.epochs, num_batches_per_epoch=self.batches_p_epoch)
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
    

##############################################################################################
##############################################################################################
######################################## 2D CNN LSTM Model ###################################
##############################################################################################
##############################################################################################


class CNN_2D_LSTM(pl.LightningModule):
    def __init__(self, cnn_input_size, n_features, hidden_size, num_layers, output_size, lookback, dropout, reduced_channel=1, bidirectional=False, lr=1e-3):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.cnn_input_size = cnn_input_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lookback = lookback
        self.dropout = dropout
        self.reduced_channel = reduced_channel
        self.bidirectional = bidirectional
        self.lr = lr

        # Feature projection which reduces the number of channels (best model still this and then lstm - although maybe should opt hyperparams and see)
        self.projections = nn.Sequential(
            nn.Conv2d(in_channels=self.cnn_input_size, out_channels=self.cnn_input_size // 2, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.cnn_input_size//2),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(in_channels=self.cnn_input_size//2, out_channels=self.reduced_channel, kernel_size=1),
            nn.ReLU(True),
        )
        # Reduce the features dimension using 1D convolutions for each channel projection
        self.conv1d_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.n_features, out_channels=64, kernel_size=1),
                nn.ReLU(True),
                nn.Dropout(self.dropout),
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
                nn.ReLU(True),
                nn.Dropout(self.dropout),
            )
            for _ in range(self.reduced_channel)  # Create separate 1D convolution layers for each channel
        ])

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=32*self.reduced_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0,  # Add dropout layer for regularization
            bidirectional=self.bidirectional,
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),  # Add dropout here
        )

    def forward(self, x):
        # Feature Projections (Channel Pooling)
        projected_x = self.projections(x)
        conv_input = projected_x.permute(0, 1, 3, 2)  # Permute (batch, channel, features, seq_len)
    
        # Apply 1D convolutions to each channel
        conv_features_list = []
        for i, conv_layer in enumerate(self.conv1d_layers):
            conv_channel = conv_input[:, i, :, :]  # Extract the i-th channel from the input
            conv_features = conv_layer(conv_channel)  # Apply convolution to the channel
            conv_features_list.append(conv_features)

        conv_features = torch.cat(conv_features_list, dim=1)
        conv_features = conv_features.permute(0, 2, 1)  # Permute (batch, seq_len, features)

        # LSTM sequence modeling
        lstm_output, _ = self.lstm(conv_features)
        lstm_output = lstm_output[:, -1, :]  # Taking the last time step's output
        
        regression_output = self.fc(lstm_output)
        return regression_output

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        # Ranger requires manual backward pass since it is designed/executed differently to base torch optimizers (Ranger doesnt work with torch compile)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=self.lr)
        return optimizer
    

##############################################################################################
##############################################################################################
################################ Multihead 1D CNN LSTM Model #################################
##############################################################################################
##############################################################################################


class MH_CNN_1D_LSTM(pl.LightningModule):
    def __init__(self, cnn_input_size, n_features, hidden_size, num_layers, output_size, lookback, dropout, epochs, batches_p_epoch, reduced_features=2, bidirectional=False, lr=1e-3, reduction_factor=0.5):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.cnn_input_size = cnn_input_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lookback = lookback
        self.dropout = dropout
        self.reduced_features = reduced_features
        self.bidirectional = bidirectional
        self.lr = lr
        self.epochs = epochs
        self.batches_p_epoch = batches_p_epoch

        # Feature selection and dimensionality reduction using 1D convolution layers on each channel
        self.conv1d_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.n_features, out_channels=64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Dropout(self.dropout),
                nn.Conv1d(in_channels=64, out_channels=self.reduced_features, kernel_size=1),
                nn.BatchNorm1d(self.reduced_features),
                nn.ReLU(True),
                nn.Dropout(self.dropout),
            )
            for _ in range(self.cnn_input_size)  # Create separate 1D convolution layers for each channel
        ])

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.cnn_input_size * self.reduced_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0,  # Add dropout only after lstm layer for regularization
            bidirectional=self.bidirectional,
        )

        # Fully connected layer
        if self.bidirectional:
            self.fc = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size * 2, self.hidden_size), # Bidirectional 2times output
                nn.ReLU(True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.output_size),
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.output_size),
            )

    def forward(self, x):
        # 1D convolution on each channel to reduce dimensionality
        x = x.permute(0, 1, 3, 2)  # Permute (batch, channel, features, seq_len)
        conv_features_list = []
        for i, conv_layer in enumerate(self.conv1d_layers):
            conv_channel = x[:, i, :, :]  # Extract the i-th channel from the input
            conv_features = conv_layer(conv_channel)  # Apply convolution to the channel
            conv_features_list.append(conv_features)

        conv_features = torch.cat(conv_features_list, dim=1)
        conv_features = conv_features.permute(0, 2, 1)  # Permute (batch, seq_len, features)

        # LSTM
        lstm_output, _ = self.lstm(conv_features)
        lstm_output = lstm_output[:, -1, :]

        # FC
        out = self.fc(lstm_output)
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        # Ranger requires manual backward pass since it is designed/executed differently to base torch optimizers (Ranger doesnt work with torch compile)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Ranger21(self.parameters(), lr=self.lr, num_epochs=self.epochs, num_batches_per_epoch=self.batches_p_epoch)
        return optimizer