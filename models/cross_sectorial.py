from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from matplotlib.lines import Line2D
from pytorch_ranger import Ranger
from ranger21 import Ranger21
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.functional import accuracy

##############################################################################################
##############################################################################################
######################################## Vanilla LSTM ########################################
##############################################################################################
##############################################################################################


class Vanilla_LSTM(pl.LightningModule):
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
                 loss_fn=F.mse_loss):
        
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()

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
                            dropout=self.dropout if self.lstm_layers > 1 else 0, 
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
                 cnn_layers=3,
                 conv_factor=0.5,
                 lstm_layers=2, 
                 lstm_nodes=64, 
                 fc_layers=1,
                 fc_nodes=64,
                 dropout=0, 
                 bidirectional=True,
                 lr=1e-3, 
                 optimizer=Ranger21,
                 activation=nn.Mish(True),
                 loss_fn=F.mse_loss):
        
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()

        self.n_companies = n_companies
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
        self.activation = activation
        self.loss_fn = loss_fn


        # Feature selection and dimensionality reduction using 1D convolutions
        cnn_modules = []
        in_channels = self.n_features
        for _ in range(self.cnn_layers):
            out_channels = int(in_channels * self.conv_factor) if int(in_channels * self.conv_factor) > 1 else 1 # Need this in 2D models as well!!!!!
            cnn_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            cnn_modules.append(self.activation)
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
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
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
################################## 2D CNN LSTM Model (proj + mh) #############################
##############################################################################################
##############################################################################################


class P_MH_CNN_2D_LSTM(pl.LightningModule):
    def __init__(self, 
                 n_companies,
                 n_features,
                 lookback,
                 epochs,
                 batches_p_epoch,
                 proj_layers=3,
                 proj_factor=0.5,
                 cnn_layers=3,
                 conv_factor=0.5,
                 lstm_layers=2, 
                 lstm_nodes=64, 
                 fc_layers=1,
                 fc_nodes=64,
                 dropout=0, 
                 bidirectional=True,
                 lr=1e-3, 
                 optimizer=Ranger21,
                 classification=False,):
        
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()

        self.n_companies = n_companies
        self.n_features = n_features
        self.lookback = lookback
        self.epochs = epochs
        self.batches_p_epoch = batches_p_epoch
        self.proj_layers = proj_layers
        self.proj_factor = proj_factor
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
        self.classification = classification


        # Feature projection which reduces the number of channels (best model still this and then lstm - although maybe should opt hyperparams and see)
        projection_modules = []
        in_channels = self.n_companies
        for _ in range(self.proj_layers):
            out_channels = int(in_channels * self.proj_factor) if int(in_channels * self.proj_factor) > 1 else 1
            projection_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            projection_modules.append(nn.ReLU(True))
            projection_modules.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        self.projections = nn.Sequential(*projection_modules)


        # Reduce the features dimension using 1D convolutions for each channel projection
        conv1d_layers = []
        in_features = self.n_features
        for _ in range(self.cnn_layers):
            out_features = int(in_features * self.conv_factor) if int(in_features * self.conv_factor) > 1 else 1
            conv1d_layers.append(nn.Conv1d(in_features, out_features, kernel_size=1))
            conv1d_layers.append(nn.ReLU(True))
            conv1d_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features

        self.conv1d_channels = nn.ModuleList([
            nn.Sequential(*conv1d_layers) for _ in range(out_channels)
        ])

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=in_features*out_channels,
            hidden_size=self.lstm_nodes,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        # Fully connected layer
        if self.fc_layers == 1:
            if self.classification:
                self.fc = nn.Linear(self.lstm_nodes*(1+self.bidirectional), self.n_companies*5)
            else:
                self.fc = nn.Linear(self.lstm_nodes*(1+self.bidirectional), self.n_companies)
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
            if self.classification:
                self.fc = nn.Sequential(*lin_layers, nn.Linear(self.fc_nodes, self.n_companies*5))
            else:
                self.fc = nn.Sequential(*lin_layers, nn.Linear(self.fc_nodes, self.n_companies))

    def forward(self, x):
        # Feature Projections (Channel Pooling)
        projected_x = self.projections(x)
        conv_input = projected_x.permute(0, 1, 3, 2)  # Permute (batch, channel, features, seq_len)
        # Apply 1D convolutions to each channel
        conv_features_list = []
        for i, conv_layer in enumerate(self.conv1d_channels):
            conv_channel = conv_input[:, i, :, :]  # Extract the i-th channel from the input
            conv_features = conv_layer(conv_channel)  # Apply convolution to the channel
            conv_features_list.append(conv_features)

        conv_features = torch.cat(conv_features_list, dim=1)
        conv_features = conv_features.permute(0, 2, 1)  # Permute (batch, seq_len, features)

        # LSTM sequence modeling
        lstm_output, _ = self.lstm(conv_features)
        lstm_output = lstm_output[:, -1, :]  # Taking the last time step's output
        
        # Classification output
        if self.classification:
            classification_output = self.fc(lstm_output).reshape(-1, self.n_companies, 5)
            # classification_output = F.softmax(classification_output, dim=2)
            return classification_output
        
        # Regression output
        regression_output = self.fc(lstm_output)
        return regression_output

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)

        if self.classification:
            loss = F.cross_entropy(y_hat, y)
            acc = accuracy(y_hat.argmax(dim=2), y.argmax(dim=2), task="multiclass", num_classes=5)
        else:
            loss = F.mse_loss(y_hat, y)
            acc = None  # No accuracy for regression
        
        self.log("train_loss", loss, prog_bar=True)
        if acc is not None:
            self.log("train_accuracy", acc, prog_bar=True)
    
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)

        if self.classification:
            loss = F.cross_entropy(y_hat, y)
            acc = accuracy(y_hat.argmax(dim=2), y.argmax(dim=2), task="multiclass", num_classes=5)
        else:
            loss = F.mse_loss(y_hat, y)
            acc = None  # No accuracy for regression
        
        self.log("val_loss", loss, prog_bar=True)
        if acc is not None:
            self.log("val_accuracy", acc, prog_bar=True)
            
        return loss
    
    def configure_optimizers(self):
        if self.optimizer == Ranger21:
            optimizer = self.optimizer(self.parameters(), lr=self.lr, num_epochs=self.epochs, num_batches_per_epoch=self.batches_p_epoch)
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer


##############################################################################################
##############################################################################################
#################################### 2D CNN LSTM Model (proj) ################################
##############################################################################################
##############################################################################################


class P_CNN_2D_LSTM(pl.LightningModule):
    def __init__(self, 
                 n_companies,
                 n_features,
                 lookback,
                 epochs,
                 batches_p_epoch,
                 proj_layers=3,
                 proj_factor=0.5,
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

        self.n_companies = n_companies
        self.n_features = n_features
        self.lookback = lookback
        self.epochs = epochs
        self.batches_p_epoch = batches_p_epoch
        self.proj_layers = proj_layers
        self.proj_factor = proj_factor
        self.lstm_layers = lstm_layers
        self.lstm_nodes = lstm_nodes
        self.fc_layers = fc_layers
        self.fc_nodes = fc_nodes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.optimizer = optimizer

        # Feature projection which reduces the number of channels (best model still this and then lstm - although maybe should opt hyperparams and see)
        projection_modules = []
        in_channels = self.n_companies
        for _ in range(self.proj_layers):
            out_channels = int(in_channels * self.proj_factor) if int(in_channels * self.proj_factor) > 1 else 1
            projection_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            projection_modules.append(nn.ReLU(True))
            projection_modules.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        self.projections = nn.Sequential(*projection_modules)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.n_features*out_channels,
            hidden_size=self.lstm_nodes,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        # Fully connected layer
        if self.fc_layers == 1:
            self.fc = nn.Linear(self.lstm_nodes*(1+self.bidirectional), self.n_companies)
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

            self.fc = nn.Sequential(*lin_layers, nn.Linear(self.fc_nodes, self.n_companies))

    def forward(self, x):
        # Feature Projections (Channel Pooling)
        projected_x = self.projections(x)
        projected_x = projected_x.reshape(projected_x.shape[0], self.lookback, -1)

        # LSTM sequence modeling
        lstm_output, _ = self.lstm(projected_x)
        lstm_output = lstm_output[:, -1, :]  # Taking the last time step's output
        
        regression_output = self.fc(lstm_output)
        return regression_output

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
        y_hat = self.forward(X)
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
##################################### 2D CNN LSTM Model (MH) #################################
##############################################################################################
##############################################################################################

    
class MH_CNN_2D_LSTM(pl.LightningModule):
    def __init__(self, 
                 n_companies,
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

        self.n_companies = n_companies
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


        # Reduce the features dimension using 1D convolutions for each channel
        conv1d_layers = []
        in_features = self.n_features
        for _ in range(self.cnn_layers):
            out_features = int(in_features * self.conv_factor) if int(in_features * self.conv_factor) > 1 else 1
            conv1d_layers.append(nn.Conv1d(in_features, out_features, kernel_size=1))
            conv1d_layers.append(nn.ReLU(True))
            conv1d_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features

        self.conv1d_channels = nn.ModuleList([
            nn.Sequential(*conv1d_layers) for _ in range(self.n_companies)
        ])

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=in_features*self.n_companies,
            hidden_size=self.lstm_nodes,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        # Fully connected layer
        if self.fc_layers == 1:
            self.fc = nn.Linear(self.lstm_nodes*(1+self.bidirectional), self.n_companies)
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

            self.fc = nn.Sequential(*lin_layers, nn.Linear(self.fc_nodes, self.n_companies))

    def forward(self, x):
        conv_input = x.permute(0, 1, 3, 2) 
        # Apply 1D convolutions to each channel
        conv_features_list = []
        for i, conv_layer in enumerate(self.conv1d_channels):
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
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
    
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        if self.optimizer == Ranger21:
            optimizer = self.optimizer(self.parameters(), lr=self.lr, num_epochs=self.epochs, num_batches_per_epoch=self.batches_p_epoch)
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
