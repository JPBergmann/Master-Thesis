import os
import random
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import sklearn
import torch
from pytorch_ranger import Ranger
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from torch import mps

from helpers.iterative import IT_DATAMODULE
from models.iterative import CNN_1D_LSTM, LSTM


def main():
    # Select random company
    with open("./DATA/Tickers/month_tickers_clean.txt", "r") as f:
        tickers = f.read().strip().split("\n")
    ticker_idx = random.choice(range(len(tickers)))
    print(f"Predicting for: {tickers[ticker_idx]} with idx: {ticker_idx}")
    # Clear GPU cache
    mps.empty_cache()
    # Set global seed for reproducibility in numpy, torch, scikit-learn
    pl.seed_everything(42)

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    LEARNING_RATE = 1e-4 # 1e-4 ind standard
    EPOCHS = 1000
    BATCH_SIZE = 32
    LOOKBACK = 12
    PRED_HORIZON = 1
    MULTISTEP = False

    data = IT_DATAMODULE(batch_size=BATCH_SIZE, lookback=LOOKBACK, pred_horizon=PRED_HORIZON, multistep=False, data_type="monthly", ticker_idx=ticker_idx, overwrite_cache=True, pred_target="return")
    data.prepare_data()
    data.setup()
    # data = CS_DATAMODULE(batch_size=BATCH_SIZE, lookback=LOOKBACK, pred_horizon=PRED_HORIZON, multistep=MULTISTEP, data_type="monthly")
    
    model = CNN_1D_LSTM(input_size=data.X_val_tensor.shape[-1], hidden_size=64, num_layers=2, output_size=1, dropout=0, lr=LEARNING_RATE)
    # model = CNN_2D_LSTM(cnn_input_size=409, lstm_input_size=159*10, hidden_size=159*2, num_layers=1, output_size=409, lookback=LOOKBACK, dropout=0)
    # model = CNN_1D_LSTM(cnn_input_size=409, lstm_input_size=159, hidden_size=128, num_layers=2, output_size=409, lookback=LOOKBACK, dropout=0)
    # compiled_model = torch.compile(model, mode="reduce-overhead", backend="aot_eager")

    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=30, mode="min")
    checkpoint = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    #compiled_model = torch.compile(model, mode="reduce-overhead", backend="aot_eager")
    mps.empty_cache()

    convlstm_trainer = pl.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, log_every_n_steps=1, callbacks=[early_stopping, checkpoint], enable_checkpointing=True, enable_progress_bar=True, default_root_dir="./lightning_logs/cnnlstm/")
    convlstm_trainer.fit(model=model, datamodule=data)
    mps.empty_cache()

    print(f"Best model path: {checkpoint.best_model_path}")
    print(f"Best model score: {checkpoint.best_model_score}")

    best_model = CNN_1D_LSTM.load_from_checkpoint(checkpoint_path=checkpoint.best_model_path).to(DEVICE)

    best_model.eval()
    with torch.inference_mode():
        y_pred = best_model(data.X_val_tensor.to(DEVICE)).cpu().detach().numpy()

    # print(f"R2 score: {r2_score(data.y_val_tensor, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(data.y_val_tensor, y_pred):.4%}")
    print(f"MAE: {mean_absolute_error(data.y_val_tensor, y_pred):.4f}")

    np.set_printoptions(suppress=True)
    print(np.round(y_pred.squeeze(), 4))
    print(np.round(data.y_val_tensor.squeeze().cpu().detach().numpy(), 4))

    # Clear GPU cache
    mps.empty_cache()

if __name__ == "__main__":
    main()
