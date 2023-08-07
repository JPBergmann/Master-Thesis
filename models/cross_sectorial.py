from typing import Any

import lightning.pytorch as pl
import torch
from pytorch_ranger import Ranger
from torch import nn, optim

##############################################################################################
##############################################################################################
###################################### 1D CNN LSTM Model #####################################
##############################################################################################
##############################################################################################


class CNN_1D_LSTM(pl.LightningModule):
    def __init__(self, cnn_input_size, lstm_input_size, hidden_size, num_layers, output_size, lookback, dropout, last_conv_size=10, lr=1e-3):
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

        # Feature selection and dimensionality reduction using stacked 2D convolution layers that ultimately result in only 1 channel
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.cnn_input_size, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
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
        out = self.cnn(x)
        out = out.reshape(out.shape[0], self.lookback, -1)
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98))    #Ranger(self.parameters(), lr=self.lr)
        return optimizer
    

##############################################################################################
##############################################################################################
######################################## 2D CNN LSTM Model ###################################
##############################################################################################
##############################################################################################


class CNN_2D_LSTM(pl.LightningModule):
    def __init__(self, cnn_input_size, lstm_input_size, hidden_size, num_layers, output_size, lookback, dropout, last_conv_size=10, lr=1e-3):
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

        # Feature selection and dimensionality reduction using stacked 2D convolution layers that ultimately result in only 1 channel
        self.cnn = nn.Sequential(
            nn.Conv2d(self.cnn_input_size, 200, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(200),
            nn.Conv2d(200, 10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Dropout(self.dropout),
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
        out = self.cnn(x)
        out = out.reshape(out.shape[0], self.lookback, -1)
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98))    #Ranger(self.parameters(), lr=self.lr)
        return optimizer
    

# ConvLSTM Cell to be used in any ConvLSTM Architectures
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

# class ConvLSTMCell(nn.Module):

#     def __init__(self, in_channels, out_channels, 
#     kernel_size, padding, activation, frame_size):

#         super(ConvLSTMCell, self).__init__()  

#         if activation == "tanh":
#             self.activation = torch.tanh 
#         elif activation == "relu":
#             self.activation = torch.relu
        
#         # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
#         self.conv = nn.Conv2d(
#             in_channels=in_channels + out_channels, 
#             out_channels=4 * out_channels, 
#             kernel_size=kernel_size, 
#             padding=padding)           

#         # Initialize weights for Hadamard Products
#         self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
#         self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
#         self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

#     def forward(self, X, H_prev, C_prev):

#         # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
#         conv_output = self.conv(torch.cat([X, H_prev], dim=1))

#         # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
#         i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

#         input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
#         forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

#         # Current Cell output
#         C = forget_gate*C_prev + input_gate * self.activation(C_conv)

#         output_gate = torch.sigmoid(o_conv + self.W_co * C )

#         # Current Hidden State
#         H = output_gate * self.activation(C)

#         return H, C
    

##############################################################################################
##############################################################################################
########################################## ConvLSTM ##########################################
##############################################################################################
##############################################################################################


class ConvLSTMLayer(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMLayer, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width) # might have to specify device
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
        height, width)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output
    

class ConvLSTM(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers):

        super(ConvLSTM, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTMLayer(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTMLayer(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:,:,-1])
        
        return output


##############################################################################################
##############################################################################################
########################################### ConvLSTM AE ######################################
##############################################################################################
##############################################################################################


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
    def __init__(self, batch_size, lookback, pred_horizon, hidden_dim=64, lr=1e-4):
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
        optimizer.step()

        # sch = self.lr_schedulers()
        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 50 == 0:
        #    sch.step()

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.squeeze()

        y_hat = self.forward(X).squeeze()
        
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, verbose=True)
        return [optimizer]#, [scheduler]
