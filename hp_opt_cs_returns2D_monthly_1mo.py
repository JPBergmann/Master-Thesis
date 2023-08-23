"""
Script that optimizes hyperparameters for 1D Data for stock returns using optuna.
"""

import os
import time

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
from models.cross_sectorial import (MH_CNN_2D_LSTM, P_CNN_2D_LSTM,
                                    P_MH_CNN_2D_LSTM)


def objective(trial):
    # Starting time and logs
    LOG_DIR = "./logs/hp_opt_cs_returns2D_monthly_1mo/"

    # Data hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lookback = trial.suggest_categorical("lookback", [1, 3, 6, 12, 24, 36])

    # Trainer hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Ranger21", "Ranger", "Adam", "SGD"])
    if optimizer_name == "Ranger21":
        optimizer = Ranger21
    elif optimizer_name == "Ranger":
        optimizer = Ranger
    elif optimizer_name == "Adam":
        optimizer = Adam
    elif optimizer_name == "SGD":
        optimizer = SGD

    # loss_name = trial.suggest_categorical("loss", ["mse", "huber", "l1"])
    # if loss_name == "mse":
    #     loss_fn = MSELoss()
    # elif loss_name == "huber":
    #     loss_fn = HuberLoss()
    # elif loss_name == "l1":
    #     loss_fn = L1Loss()

    activation_name = trial.suggest_categorical("activation", ["relu", "lrelu", "tanh"])
    if activation_name == "relu":
        activation = ReLU(True)
    elif activation_name == "lrelu":
        activation = LeakyReLU(True)

    scaler_name = trial.suggest_categorical("scaler", ["standard", "minmax", "robust"])
    if scaler_name == "standard":
        scaler = scale
    elif scaler_name == "minmax":
        scaler = minmax_scale
    elif scaler_name == "robust":
        scaler = robust_scale

    # Data module 
    data = CS_DATAMODULE_2D(
        batch_size=batch_size,
        lookback=lookback,
        pred_horizon=1,
        multistep=False,
        data_type="monthly",
        train_workers=0,
        overwrite_cache=False,
        pred_target="return",
        scaling_fn=scaler,
    )

    # Global model hyperparameters and constants
    data.prepare_data()
    data.setup()

    N_COMPANIES = int(len(data.tickers))
    N_FEATURES = int(data.X_train_tensor.shape[-1])
    EPOCHS = trial.suggest_categorical("epochs", [500, 1000, 5000])
    N_BATCHES = int(np.ceil(len(data.X_train_tensor) / batch_size))

    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    dropout = trial.suggest_float("dropout", 0, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 6)
    lstm_nodes = trial.suggest_categorical("lstm_nodes", [32, 64, 128, 256, 512])
    fc_layers = trial.suggest_int("fc_layers", 1, 6)
    fc_nodes = trial.suggest_categorical("fc_nodes", [32, 64, 128, 256, 512])


    proj_layers = trial.suggest_int("proj_layers", 2, 6)
    proj_factor = trial.suggest_float("proj_factor", 0.1, 0.8)
    cnn_layers = trial.suggest_int("cnn_layers", 2, 6)
    conv_factor = trial.suggest_float("conv_factor", 0.1, 0.8)

    model = P_MH_CNN_2D_LSTM(
        n_companies=N_COMPANIES,
        n_features=N_FEATURES,
        lookback=lookback,
        epochs=EPOCHS,
        batches_p_epoch=N_BATCHES,
        proj_layers=proj_layers,
        proj_factor=proj_factor,
        cnn_layers=cnn_layers,
        conv_factor=conv_factor,
        lstm_layers=lstm_layers,
        lstm_nodes=lstm_nodes,
        fc_layers=fc_layers,
        fc_nodes=fc_nodes,
        dropout=dropout,
        bidirectional=bidirectional,
        lr=lr,
        optimizer=optimizer,
        activation=activation,
        loss_fn=mse_loss,
    )


    swa = StochasticWeightAveraging(1e-2)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    logger = TensorBoardLogger(save_dir=LOG_DIR, name=f"trial_{trial.number}", version=0)
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")


    # Set global seed for reproducibility in numpy, torch, scikit-learn
    pl.seed_everything(42)

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

    trainer = pl.Trainer(
        #default_root_dir=os.path.join(LOG_DIR, f"trial_{trial.number}"),
        accelerator=DEVICE,
        devices=1,
        logger=logger,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        callbacks=[swa, early_stopping, checkpoint, pruner],
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    trainer.fit(model, data)

    with open(os.path.join(LOG_DIR, f"trial_{trial.number}", "best_checkpoint.txt"), "w") as f:
        f.write(f"Model: {model._get_name()}\n")
        f.write(f"Best Checkpoint: {checkpoint.best_model_path}\n")

    return trainer.callback_metrics["val_loss"].item()


# Run the Optuna study
if __name__ == "__main__":

    prune = MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=prune)
    study.optimize(objective, timeout=259_200, catch=(UserWarning, RuntimeError)) # Timeout of 3 days (259200 seconds)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    study.best_trial
    print("Bets trial num:")
    study.best_trial.number
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))
