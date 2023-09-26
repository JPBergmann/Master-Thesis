"""
Script that optimizes hyperparameters for 2D cs Data for stock prices using optuna.
"""

import logging
import os
import sys
import time
import shutil

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import sklearn
import torch
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner
from pytorch_lightning.callbacks import (  # old import to circumvent optuna bug
    EarlyStopping, ModelCheckpoint, StochasticWeightAveraging)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_ranger import Ranger
from ranger21 import Ranger21
from sklearn.preprocessing import minmax_scale, robust_scale, scale
from torch import cuda, mps
from torch.nn import HuberLoss, L1Loss, LeakyReLU, Mish, MSELoss, ReLU, Tanh
from torch.nn.functional import mse_loss
from torch.optim import SGD, Adam

from helpers.cross_sectorial import CS_DATAMODULE_2D
from models.cross_sectorial import P_CNN_2D_LSTM


def objective(trial):
    # Starting time and logs
    LOG_DIR = f"./logs/hp_opt_cs_price_monthly_1mo_cluster0batch1v2"

    # Delete all logs except that of the best trial
    if os.path.exists(LOG_DIR):
        for file in os.listdir(LOG_DIR):
            if file != f"trial_{study.best_trial.number}":
                shutil.rmtree(os.path.join(LOG_DIR, file))

    # Data hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    lookback = trial.suggest_categorical("lookback", [1, 3, 6, 12, 24, 36])

    # Trainer hyperparameters
    optimizer = Ranger

    # Data module 
    data = CS_DATAMODULE_2D(
        batch_size=batch_size,
        lookback=lookback,
        pred_horizon=1,
        multistep=False,
        data_type="monthly",
        train_workers=0,
        overwrite_cache=False,
        pred_target="price",
        scaling_fn=robust_scale,
        cluster=0,
        goal="regression"
    )

    # Global model hyperparameters and constants
    data.prepare_data()
    data.setup()

    N_COMPANIES = 136 #int(len(data.tickers))
    N_FEATURES = int(data.X_train_tensor.shape[-1])
    EPOCHS = trial.suggest_categorical("epochs", [500, 1000, 5000])
    N_BATCHES = int(np.ceil(len(data.X_train_tensor) / batch_size))

    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout = trial.suggest_float("dropout", 0, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
    lstm_nodes = trial.suggest_categorical("lstm_nodes", [64, 128, 256, 512, 768, 1024])
    fc_layers = trial.suggest_int("fc_layers", 1, 4)
    fc_nodes = trial.suggest_categorical("fc_nodes", [64, 128, 256, 512, 768, 1024])
    proj_layers = trial.suggest_int("proj_layers", 1, 4)
    proj_factor = trial.suggest_float("proj_factor", 0.2, 0.5)

    model = P_CNN_2D_LSTM(
        n_companies=N_COMPANIES,
        n_features=N_FEATURES,
        lookback=lookback,
        epochs=EPOCHS,
        batches_p_epoch=N_BATCHES,
        proj_layers=proj_layers,
        proj_factor=proj_factor,
        lstm_layers=lstm_layers,
        lstm_nodes=lstm_nodes,
        fc_layers=fc_layers,
        fc_nodes=fc_nodes,
        dropout=dropout,
        bidirectional=bidirectional,
        lr=lr,
        optimizer=optimizer,
    )


    swa = StochasticWeightAveraging(1e-2)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    logger = TensorBoardLogger(save_dir=LOG_DIR, name=f"trial_{trial.number}", version=0)
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")


    # Set global seed for reproducibility in numpy, torch, scikit-learn
    pl.seed_everything(42)

    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu = cuda
        torch.set_float32_matmul_precision("medium")
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        gpu = mps
    else:
        DEVICE = "cpu"
        gpu = None

    trainer = pl.Trainer(
        #default_root_dir=os.path.join(LOG_DIR, f"trial_{trial.number}"),
        accelerator=DEVICE,
        devices=1,
        logger=logger,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        callbacks=[swa, early_stopping, checkpoint, pruner],
        log_every_n_steps=1,
    )

    trainer.fit(model, data)
    gpu.empty_cache()

    with open(os.path.join(LOG_DIR, f"trial_{trial.number}", "best_checkpoint.txt"), "w") as f:
        f.write(f"Model: {model._get_name()}\n")
        f.write(f"Best Checkpoint: {checkpoint.best_model_path}\n")

    return trainer.callback_metrics["val_loss"].item()


# Run the Optuna study
if __name__ == "__main__":

    LOG_DIR = f"./logs/hp_opt_cs_price_monthly_1mo_cluster0batch1v2"

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    prune = MedianPruner(n_startup_trials=50, n_warmup_steps=100, interval_steps=5)
    study = optuna.create_study(direction="minimize", pruner=prune)
    study.optimize(objective, n_trials=500, catch=(UserWarning, RuntimeError, ValueError, MemoryError)) # Run for 36 hours

    with open(os.path.join(LOG_DIR, "best_trial.txt"), "w") as f:
        f.write(f"trial_{study.best_trial.number}\n")
        f.write(f"Loss: {study.best_trial.value}\n")
        f.write(f"Params: {study.best_trial.params}\n")
