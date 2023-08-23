from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_ranger import Ranger
from ranger21 import Ranger21
from torch import nn, optim

##############################################################################################
##############################################################################################
######################################## Vanilla LSTM ########################################
##############################################################################################
##############################################################################################


class LSTM(pl.LightningModule):
    def __init__(self, 
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
                optimizer=Ranger21,):
            
            super().__init__()

            self.automatic_optimization = False
            self.save_hyperparameters()


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

            self.lstm = nn.LSTM(input_size=self.n_features, 
                                hidden_size=self.lstm_nodes, 
                                num_layers=self.lstm_layers, 
                                batch_first=True, 
                                dropout=self.dropout if self.lstm_layers > 1 else 0, 
                                bidirectional=self.bidirectional)

            if self.fc_layers == 1:
                self.fc = nn.Linear(self.lstm_nodes*(1+self.bidirectional), 1)
            else:
                lin_layers = []
                for i in range(self.fc_layers):

                    if i == 0:
                        lin_layers.append(nn.Linear(self.lstm_nodes*(1+self.bidirectional), self.fc_nodes))
                        lin_layers.append(nn.ReLU(True))
                        lin_layers.append(nn.Dropout(self.dropout))
                    else:
                        lin_layers.append(nn.Linear(self.fc_nodes, self.fc_nodes))
                        lin_layers.append(nn.ReLU(True))
                        lin_layers.append(nn.Dropout(self.dropout))

                self.fc = nn.Sequential(*lin_layers, nn.Linear(self.fc_nodes, 1))


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = F.mse_loss(y_hat, y)
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
        loss = F.mse_loss(y_hat, y)
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
######################################## 1D CNN LSTM #########################################
##############################################################################################
##############################################################################################


class CNN_1D_LSTM(pl.LightningModule):
    def __init__(self, 
                 n_features,
                 lookback,
                 epochs,
                 batches_p_epoch,
                 cnn_layers=3,
                 conv_factor=0.5,
                 lstm_layers=2, 
                 lstm_nodes=64, 
                 fc_layers=1,
                 fc_nodes=64,
                 dropout=0, 
                 bidirectional=True,
                 lr=1e-3, 
                 optimizer=Ranger21,):
        
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()


        self.n_features = n_features
        self.lookback = lookback
        self.epochs = epochs
        self.batches_p_epoch = batches_p_epoch
        self.cnn_layers = cnn_layers
        self.conv_factor = conv_factor
        self.lstm_layers = lstm_layers
        self.lstm_nodes = lstm_nodes
        self.fc_layers = fc_layers
        self.fc_nodes = fc_nodes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.optimizer = optimizer


        # Feature selection and dimensionality reduction using 1D convolutions
        cnn_modules = []
        in_channels = self.n_features
        for _ in range(self.cnn_layers):
            out_channels = int(in_channels * self.conv_factor) if int(in_channels * self.conv_factor) > 1 else 1
            cnn_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            cnn_modules.append(nn.ReLU(True))
            cnn_modules.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_modules)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.lstm_nodes,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        # Fully connected layer
        if self.fc_layers == 1:
            self.fc = nn.Linear(self.lstm_nodes*(1+self.bidirectional), 1)
        else:
            lin_layers = []
            for i in range(self.fc_layers):

                if i == 0:
                    lin_layers.append(nn.Linear(self.lstm_nodes*(1+self.bidirectional), self.fc_nodes))
                    lin_layers.append(nn.ReLU(True))
                    lin_layers.append(nn.Dropout(self.dropout))
                else:
                    lin_layers.append(nn.Linear(self.fc_nodes, self.fc_nodes))
                    lin_layers.append(nn.ReLU(True))
                    lin_layers.append(nn.Dropout(self.dropout))

            self.fc = nn.Sequential(*lin_layers, nn.Linear(self.fc_nodes, 1))


    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        out = out.permute(0, 2, 1)
        # LSTM
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X).squeeze(1)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        if self.optimizer == Ranger21:
            optimizer = self.optimizer(self.parameters(), lr=self.lr, num_epochs=self.epochs, num_batches_per_epoch=self.batches_p_epoch)
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
