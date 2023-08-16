import os
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import sklearn
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from lightning.pytorch.loggers import WandbLogger
from pytorch_ranger import Ranger
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from torch import cuda, mps

from helpers.cross_sectorial import (CS_DATAMODULE_1D, CS_DATAMODULE_2D,
                                     CS_VID_DATAMODULE)
from models.cross_sectorial import (CNN_1D_LSTM, CNN_2D_LSTM, MH_CNN_1D_LSTM,
                                    ConvLSTM_AE)


def main():

    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu = cuda
        # torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        gpu = mps
    else:
        DEVICE = "cpu"
        gpu = None

    # DEVICE = "cpu"
    # Clear GPU cache
    if gpu:
        gpu.empty_cache()
    # Set global seed for reproducibility in numpy, torch, scikit-learn
    pl.seed_everything(42)

    LEARNING_RATE = 1e-4 # 1e-4 ind standard
    EPOCHS = 50
    BATCH_SIZE = 32
    LOOKBACK = 24
    PRED_HORIZON = 1
    MULTISTEP = False
    TRAIN_WORKERS = 0 # 0 fastest ...
    CLUSTER = 1

    # data = CS_VID_DATAMODULE(batch_size=BATCH_SIZE, lookback=LOOKBACK, pred_horizon=PRED_HORIZON, multistep=MULTISTEP, data_type="monthly", resize=None, overwrite_cache=True, pred_target="return", train_workers=TRAIN_WORKERS)
    data = CS_DATAMODULE_2D(batch_size=BATCH_SIZE, 
                         lookback=LOOKBACK, 
                         pred_horizon=PRED_HORIZON, 
                         multistep=MULTISTEP, 
                         data_type="monthly", 
                         pred_target="price", 
                         overwrite_cache=False,
                         cluster=CLUSTER)
    # data = CS_DATAMODULE_1D(batch_size=32, lookback=12, pred_horizon=1, multistep=False, data_type="monthly", pred_target="price", overwrite_cache=False, cluster=None, umap_dim=2)
    
    data.prepare_data()
    data.setup()

    N_COMPANIES = int(len(data.tickers))
    N_FEATURES = int(data.X_train_tensor.shape[-1])
    # Batches per epoch which have to be rounded up to the next integer
    N_BATCHES = int(np.ceil(len(data.X_train_tensor) / BATCH_SIZE))
    print(f"Number of batches per Epoch: {N_BATCHES}")

    # model = ConvLSTM_AE(batch_size=BATCH_SIZE, lookback=LOOKBACK, pred_horizon=1, hidden_dim=128, epochs=EPOCHS, batches_p_epoch=N_BATCHES) # Pred 1 not 21 since 1 Frame pred!!!!!! (only > 1 if data prep is multistep)
    model = MH_CNN_1D_LSTM(cnn_input_size=N_COMPANIES,  # CNN_2D_LSTM
                        n_features=N_FEATURES, 
                        hidden_size=64, 
                        num_layers=2, 
                        output_size=N_COMPANIES, 
                        lookback=LOOKBACK, 
                        dropout=0,
                        reduced_features=4, 
                        bidirectional=True,
                        lr=LEARNING_RATE,
                        epochs=EPOCHS,
                        batches_p_epoch=N_BATCHES,)
    # model = CNN_1D_LSTM(cnn_input_size=818, lstm_input_size=256, hidden_size=512, num_layers=2, output_size=409, lookback=12, dropout=0)
    # model = CNN_1D_LSTM(cnn_input_size=409, lstm_input_size=159, hidden_size=128, num_layers=2, output_size=409, lookback=LOOKBACK, dropout=0)
    # compiled_model = torch.compile(model, mode="reduce-overhead", backend="aot_eager")

    early_stopping = EarlyStopping(monitor="val_loss", patience=200, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    swa = StochasticWeightAveraging(1e-2)
    lr_logger = LearningRateMonitor(logging_interval="step")

    wandb_logger = WandbLogger(project="cross_sectorial", log_model="all")
    wandb_logger.watch(model, log="all", log_freq=1)

    trainer = pl.Trainer(accelerator=DEVICE, 
                         max_epochs=EPOCHS, 
                         log_every_n_steps=1, 
                         callbacks=[early_stopping, checkpoint_callback, swa, lr_logger], 
                         enable_checkpointing=True, 
                         enable_progress_bar=True, 
                         logger=wandb_logger, 
                         detect_anomaly=True,)
    
    trainer.fit(model=model, datamodule=data)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score}")

    best_model = MH_CNN_1D_LSTM.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path).to(DEVICE)

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
    if gpu:
        gpu.empty_cache()

if __name__ == "__main__":
    main()
