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
from pytorch_ranger import Ranger
from ranger21 import Ranger21
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from torch import cuda, mps
from torch.nn import Mish

from helpers.cross_sectorial import (CS_DATAMODULE_1D, CS_DATAMODULE_2D,)
from models.cross_sectorial import (CNN_1D_LSTM, CNN_2D_LSTM, MH_CNN_1D_LSTM,
                                    LSTM)

import warnings
warnings.filterwarnings("ignore")


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

    # DEVICE = "cpu"
    # Clear GPU cache
    if gpu:
        gpu.empty_cache()
    # Set global seed for reproducibility in numpy, torch, scikit-learn
    pl.seed_everything(42)

    LEARNING_RATE = 1e-4 # 1e-4 ind standard
    EPOCHS = 1000
    BATCH_SIZE = 32
    LOOKBACK = 12
    PRED_HORIZON = 1
    MULTISTEP = False
    TRAIN_WORKERS = 0 # 0 fastest ...
    CLUSTER = 0

    data = CS_DATAMODULE_1D(batch_size=BATCH_SIZE,
                            lookback=LOOKBACK,
                            pred_horizon=PRED_HORIZON,
                            multistep=MULTISTEP,
                            data_type="monthly",
                            train_workers=TRAIN_WORKERS,
                            overwrite_cache=False,
                            pred_target="price",
                            cluster=CLUSTER,
                            only_prices=False,
                            umap_dim=2)
    
    data.prepare_data()
    data.setup()

    N_COMPANIES = int(len(data.tickers))
    N_FEATURES = int(data.X_train_tensor.shape[-1])
    # Batches per epoch which have to be rounded up to the next integer
    N_BATCHES = int(np.ceil(len(data.X_train_tensor) / BATCH_SIZE))
    print(f"Number of batches per Epoch: {N_BATCHES}")

    model = LSTM(n_companies=N_COMPANIES,
                 n_features=N_FEATURES,
                 lookback=LOOKBACK,
                 epochs=EPOCHS,
                 batches_p_epoch=N_BATCHES,
                 lstm_layers=2,
                 lstm_nodes=512,
                 fc_layers=1,
                 fc_nodes=N_COMPANIES,
                 dropout=0.3,
                 bidirectional=True,
                 lr=LEARNING_RATE,
                 optimizer=Ranger21,
                 activation=Mish(True),)


    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    swa = StochasticWeightAveraging(1e-2)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(accelerator=DEVICE, 
                         max_epochs=EPOCHS, 
                         callbacks=[early_stopping, checkpoint_callback, swa, lr_monitor], 
                         enable_progress_bar=True, 
                         detect_anomaly=True,)
    
    trainer.fit(model=model, datamodule=data)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score}")

    best_model = LSTM.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path).to(DEVICE)

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
