from typing import Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from pytorch_ranger import Ranger
from ranger21 import Ranger21
from torch import nn, optim
from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_packed_sequence)

import wandb

##############################################################################################
##############################################################################################
###################################### 1D CNN LSTM Model #####################################
##############################################################################################
##############################################################################################


class CNN_1D_LSTM(pl.LightningModule):
    def __init__(self, cnn_input_size, lstm_input_size, hidden_size, num_layers, output_size, lookback, dropout, last_conv_size=10, lr=1e-3, optimizer=Ranger):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.cnn_input_size = cnn_input_size
        self.lstm_input_size = lstm_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lookback = lookback
        self.dropout = dropout
        self.last_conv_size = last_conv_size
        self.lr = lr
        self.optimizer = optimizer

        # Feature selection and dimensionality reduction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.cnn_input_size, out_channels=self.cnn_input_size//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.cnn_input_size//2, out_channels=self.lstm_input_size, kernel_size=1),
            nn.ReLU(),
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0,  # Add dropout layer for regularization
            bidirectional=False,
        )

        # Fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
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


##############################################################################################
##############################################################################################
########################################### ConvLSTM AE ######################################
##############################################################################################
##############################################################################################

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return plt
    

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        # outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs


class ConvLSTM_AE(pl.LightningModule):
    def __init__(self, batch_size, lookback, pred_horizon, epochs, batches_p_epoch, hidden_dim=64, lr=1e-4,):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.lookback = lookback
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batches_p_epoch = batches_p_epoch
        self.model = EncoderDecoderConvLSTM(nf=self.hidden_dim, in_chan=1)


    def forward(self, x):

        output = self.model(x, future_seq=self.pred_horizon)

        return output

    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze()

        y_hat = self.forward(X).squeeze()

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

        # Ranger requires manual backward pass since it is designed/executed differently to base torch optimizers (Ranger doesnt work with torch compile)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.3)
        chart = plot_grad_flow(self.named_parameters())
        wandb.log({"grad_flow": wandb.Image(chart)})
        optimizer.step()

        # sch = self.lr_schedulers()
        # sch.step()

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze()

        y_hat = self.forward(X).squeeze()
        
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Ranger21(self.parameters(), lr=self.lr, num_epochs=self.epochs, num_batches_per_epoch=self.batches_p_epoch)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=1e-2)
        return optimizer
