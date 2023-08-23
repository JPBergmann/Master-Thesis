"""
Script that optimizes hyperparameters for 1D Data for stock returns using optuna.
"""

import os

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
from torch.optim import SGD, Adam

from helpers.cross_sectorial import CS_DATAMODULE_1D
from models.cross_sectorial import CNN_1D_LSTM, Vanilla_LSTM


def objective(trial):
    # Data hyperparameters
    data_type = trial.suggest_categorical("data_type", ["monthly", "daily"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    multistep = trial.suggest_categorical("multistep", [True, False])
    red_each_dim = trial.suggest_categorical("red_each_dim", [None, 1, 2, 4, 6])
    red_total_dim = trial.suggest_categorical("red_total_dim", [None, 10, 25, 50, 75, 100])

    # if data_type == "monthly":
        # pred_horizon = 1
        # if (not multistep):
            # pred_horizon = trial.suggest_categorical("pred_horizon", [1, 2])
        # else:
            # pred_horizon = 2
    lookback = trial.suggest_categorical("lookback", [1, 3, 6, 12, 24, 36, 48])
    # else:
        # pred_horizon = trial.suggest_categorical("pred_horizon", [22, 41]) # Data prep stuff requires dis
        # pred_horizon = 22
        # lookback = trial.suggest_categorical("lookback", [1, 3, 6, 12, 24, 36, 48])
        # lookback = lookback * 21 # Average of 21 trading days per month (252 a year)

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

    loss_name = trial.suggest_categorical("loss", ["mse", "huber", "l1"])
    if loss_name == "mse":
        loss_fn = MSELoss()
    elif loss_name == "huber":
        loss_fn = HuberLoss()
    elif loss_name == "l1":
        loss_fn = L1Loss()

    activation_name = trial.suggest_categorical("activation", ["relu", "lrelu", "tanh"])
    if activation_name == "relu":
        activation = ReLU(True)
    elif activation_name == "lrelu":
        activation = LeakyReLU(True)
    elif activation_name == "tanh":
        activation = Tanh()

    scaler_name = trial.suggest_categorical("scaler", ["standard", "minmax", "robust"])
    if scaler_name == "standard":
        scaler = scale
    elif scaler_name == "minmax":
        scaler = minmax_scale
    elif scaler_name == "robust":
        scaler = robust_scale

    # Prune combinations that are not possible (in case something not cought before)
    # if multistep and (pred_horizon==1):
        # raise optuna.exceptions.TrialPruned("Trial pruned due to invalid combination of multistep and pred_horizon")
    
    if (red_each_dim is not None) and (red_total_dim is not None):
        raise optuna.exceptions.TrialPruned("Trial pruned due to invalid combination of red_each_dim and red_total_dim")
    elif (red_each_dim is None) and (red_total_dim is None):
        raise optuna.exceptions.TrialPruned("Trial pruned due to invalid combination of red_each_dim and red_total_dim")

    # Data module
    data = CS_DATAMODULE_1D(
        batch_size=batch_size,
        lookback=lookback,
        pred_horizon=1,
        multistep=False,
        data_type="monthly",
        train_workers=0,
        overwrite_cache=False,
        pred_target="price",
        red_each_dim=red_each_dim,
        red_total_dim=red_total_dim,
        scaling_fn=scaler,
    )

    # Global model hyperparameters and constants
    data.prepare_data()
    data.setup()

    N_COMPANIES = int(len(data.tickers))
    N_FEATURES = int(data.X_train_tensor.shape[-1])
    EPOCHS = trial.suggest_categorical("epochs", [500, 1000, 5000, 10_000])
    N_BATCHES = int(np.ceil(len(data.X_train_tensor) / batch_size))

    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    dropout = trial.suggest_float("dropout", 0, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 6)
    lstm_nodes = trial.suggest_categorical("lstm_nodes", [32, 64, 128, 256, 512])
    fc_layers = trial.suggest_int("fc_layers", 1, 6)
    fc_nodes = trial.suggest_categorical("fc_nodes", [32, 64, 128, 256, 512])

    # Model and local hyperparameters
    model_name = trial.suggest_categorical("model_name", ["LSTM", "CNN_1D_LSTM"])

    if model_name == "LSTM":
        model = Vanilla_LSTM(
            n_companies=N_COMPANIES,
            n_features=N_FEATURES,
            lookback=lookback,
            epochs=EPOCHS,
            batches_p_epoch=N_BATCHES,
            lstm_layers=lstm_layers,
            lstm_nodes=lstm_nodes,
            fc_layers=fc_layers,
            fc_nodes=fc_nodes,
            dropout=dropout,
            bidirectional=bidirectional,
            lr=lr,
            optimizer=optimizer,
            activation=activation,
            loss_fn=loss_fn,
        )

    elif model_name == "CNN_1D_LSTM":
        cnn_layers = trial.suggest_int("cnn_layers", 1, 6)
        conv_factor = trial.suggest_float("conv_factor", 0.5, 1.5)

        model = CNN_1D_LSTM(
            n_companies=N_COMPANIES,
            n_features=N_FEATURES,
            lookback=lookback,
            epochs=EPOCHS,
            batches_p_epoch=N_BATCHES,
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
            loss_fn=loss_fn,
        )

    pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    swa = StochasticWeightAveraging(1e-2)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    logger = TensorBoardLogger(save_dir=LOG_DIR, name=f"trial_{trial.number}", version=0) #os.path.join(LOG_DIR, f"trial_{trial.number}"), 


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

    # Clear GPU cache pre-training
    if gpu:
        gpu.empty_cache()

    trainer = pl.Trainer(
        #default_root_dir=os.path.join(LOG_DIR, f"trial_{trial.number}"),
        accelerator=DEVICE,
        devices=1,
        logger=logger,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        callbacks=[swa, early_stopping, pruning, checkpoint],
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    trainer.fit(model, data)

    with open(os.path.join(LOG_DIR, f"trial_{trial.number}", "best_checkpoint.txt"), "w") as f:
        f.write(f"Model: {model._get_name()}\n")
        f.write(f"Best Checkpoint: {checkpoint.best_model_path}\n")
    # Clear GPU cache post-training
    if gpu:
        gpu.empty_cache()

    return trainer.callback_metrics["val_loss"].item()


# Run the Optuna study
if __name__ == "__main__":

    LOG_DIR = "./logs/hp_opt_cs_prices_monthly_1mo/"
    SAVE_BEST_DIR = "./tuned_models/"

    pruner = MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    study.best_trial
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))
