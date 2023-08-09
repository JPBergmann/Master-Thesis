import os
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
from torch import mps, cuda

from helpers.cross_sectorial import CS_DATAMODULE_2D, CS_VID_DATAMODULE
from models.cross_sectorial import (CNN_1D_LSTM, CNN_2D_LSTM, ConvLSTM_AE)


def main():

    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu = cuda
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        gpu = mps
    else:
        DEVICE = "cpu"
        gpu = None

    # Clear GPU cache
    gpu.empty_cache()
    # Set global seed for reproducibility in numpy, torch, scikit-learn
    pl.seed_everything(42)

    LEARNING_RATE = 8e-4 # 1e-4 ind standard
    EPOCHS = 10_000
    BATCH_SIZE = 32
    LOOKBACK = 12
    PRED_HORIZON = 1
    MULTISTEP = False
    TRAIN_WORKERS = 0 # 0 fastest ...
    CLUSTER = 0

    # data = CS_VID_DATAMODULE(batch_size=BATCH_SIZE, lookback=LOOKBACK, pred_horizon=PRED_HORIZON, multistep=MULTISTEP, data_type="monthly", resize=None, overwrite_cache=True, pred_target="return", train_workers=TRAIN_WORKERS)
    data = CS_DATAMODULE_2D(batch_size=BATCH_SIZE, 
                         lookback=LOOKBACK, 
                         pred_horizon=PRED_HORIZON, 
                         multistep=MULTISTEP, 
                         data_type="monthly", 
                         pred_target="price", 
                         overwrite_cache=False,
                         cluster=CLUSTER)
    
    data.prepare_data()
    data.setup()

    N_COMPANIES = int(len(data.tickers))
    N_FEATURES = int(data.X_train_tensor.shape[-1])

    # model = ConvLSTM_AE(batch_size=BATCH_SIZE, lookback=LOOKBACK, pred_horizon=1, hidden_dim=64) # Pred 1 not 21 since 1 Frame pred!!!!!! (only > 1 if data prep is multistep)
    model = CNN_2D_LSTM(cnn_input_size=N_COMPANIES, 
                        n_features=N_FEATURES, 
                        hidden_size=1028, 
                        num_layers=1, 
                        output_size=N_COMPANIES, 
                        lookback=LOOKBACK, 
                        dropout=0.2, 
                        reduced_channel=50, 
                        bidirectional=True,
                        lr=LEARNING_RATE)
    # model = CNN_1D_LSTM(cnn_input_size=409, lstm_input_size=159, hidden_size=128, num_layers=2, output_size=409, lookback=LOOKBACK, dropout=0)
    # compiled_model = torch.compile(model, mode="reduce-overhead", backend="aot_eager")

    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    trainer = pl.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, log_every_n_steps=1, callbacks=[early_stopping, checkpoint_callback], enable_checkpointing=True, enable_progress_bar=True, default_root_dir="./lightning_logs/convae/")
    trainer.fit(model=model, datamodule=data)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score}")

    best_model = CNN_2D_LSTM.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path).to(DEVICE)

    best_model.eval()
    with torch.inference_mode():
        y_pred = best_model(data.X_val_tensor.to(DEVICE)).cpu().detach().numpy()

    print(f"R2 score: {r2_score(data.y_val_tensor.squeeze(), y_pred.squeeze()):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(data.y_val_tensor.squeeze(), y_pred.squeeze()):.4%}")
    print(f"MAE: {mean_absolute_error(data.y_val_tensor.squeeze(), y_pred.squeeze()):.4f}")

    np.set_printoptions(suppress=True)
    print(np.round(y_pred.squeeze(), 2))
    print(np.round(data.y_val_tensor.squeeze().cpu().detach().numpy(), 2))

    # Clear GPU cache
    gpu.empty_cache()

if __name__ == "__main__":
    main()
