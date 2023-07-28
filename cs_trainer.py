from typing import Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from pytorch_ranger import Ranger
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   RobustScaler, StandardScaler, minmax_scale,
                                   power_transform, scale)
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from helpers.cross_sectorial import *

# from tqdm import tqdm



# Set global seed for reproducibility in numpy, torch, scikit-learn
pl.seed_everything(42)
# torch.manual_seed(42)
# torch.mps.manual_seed(42)
# torch.backends.mps.deterministic = True
# torch.cuda.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# np.random.seed(42)


# Get data by going to project root using pd.read_parquet
data = pd.read_parquet("./DATA/Monthly/Processed/month_data_fin_tec.parquet")
macro = pd.read_parquet("./DATA/Monthly/Processed/month_data_macro_USCA.parquet")
with open("./DATA/Tickers/month_tickers_clean.txt", "r") as f:
    tickers = f.read().strip().split("\n")

X_train, X_val, X_test, y_train, y_val, y_test = format_tensors_cs(data,
                                                                macro, 
                                                                tickers[:100],
                                                                lookback=12, 
                                                                pred_horizon=1,
                                                                multistep=False,)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X_train_tensor = torch.from_numpy(X_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).float()
y_val_tensor = torch.from_numpy(y_val).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

LEARING_RATE = 1e-4 # 1e-4 ind standard
EPOCHS = 10_000
BATCH_SIZE = 64 # Small batch size since we are using a small dataset

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Num workers might make sense for daily data since more batches. 
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()

class CNN_LSTM(pl.LightningModule):
    def __init__(self, cnn_input_size, lstm_input_size, hidden_size, num_layers, output_size, lookback, dropout):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback = lookback

        # Feature selection and dimensionality reduction using stacked 2D convolution layers that ultimately result in only 1 channel
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_input_size, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(dropout),  # Add dropout layer for regularization
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,  # Add dropout layer for regularization
            bidirectional=False,
        )

        # Fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN
        out = self.cnn(x)
        out = out.reshape(out.shape[0], self.lookback, -1)
        # # # LSTM
        out, _ = self.lstm(out)
        out = self.fc2(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
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
        y_hat = self.forward(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=LEARING_RATE)
        return optimizer

cnn_input_size = X_train_tensor.shape[1]
lstm_input_size = 4992 #X_train_tensor.shape[3] * 10
hidden_size = lstm_input_size * 2 #X_train_tensor.shape[3] * 10
lookback = X_train_tensor.shape[2]
num_layers = 2 # More layers until now counterproductive
output_size = 100 # 1 if multi_step set to false, 2 for true
dropout = 0#.5 #.5

model = CNN_LSTM(cnn_input_size=cnn_input_size, 
                 lstm_input_size=lstm_input_size,
                 hidden_size=hidden_size, 
                 num_layers=num_layers, 
                 output_size=output_size,
                 lookback=lookback, 
                 dropout=dropout,)

early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

trainer = pl.Trainer(accelerator="gpu", max_epochs=EPOCHS, log_every_n_steps=1, callbacks=[early_stopping, checkpoint_callback], enable_checkpointing=True, enable_progress_bar=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

print(f"Best model path: {checkpoint_callback.best_model_path}")
print(f"Best model score: {checkpoint_callback.best_model_score}")

best_model = CNN_LSTM.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)

best_model.eval()
with torch.inference_mode():
    y_pred = best_model(X_val_tensor.to(device)).cpu().detach().numpy()

print(f"R2 score: {r2_score(y_val_tensor.squeeze(), y_pred.squeeze()):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_val_tensor, y_pred):.4%}")
print(f"MAE: {mean_absolute_error(y_val_tensor, y_pred):.4f}")

print(y_pred)
print(y_val)