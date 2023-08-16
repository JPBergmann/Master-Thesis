from typing import Any

import lightning.pytorch as pl
import torch
from pytorch_ranger import Ranger
from torch import nn, optim

##############################################################################################
##############################################################################################
######################################## Vanilla LSTM ########################################
##############################################################################################
##############################################################################################


class LSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, lr):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)#.squeeze(1)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        # Ranger requires manual backward pass since it is designed/executed differently to base torch optimizers
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X).squeeze(1)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=self.lr)
        return optimizer
    

##############################################################################################
##############################################################################################
######################################## 1D CNN LSTM #########################################
##############################################################################################
##############################################################################################


class CNN_1D_LSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, lr):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.lr = lr

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        #cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)#.squeeze(1)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        # Ranger requires manual backward pass since it is designed/executed differently to base torch optimizers
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X).squeeze(1)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=self.lr)
        return optimizer
    

##############################################################################################
##############################################################################################
########################################## ConvLSTM ##########################################
##############################################################################################
##############################################################################################