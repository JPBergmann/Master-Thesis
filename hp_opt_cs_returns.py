"""
Script that optimizes hyperparameters for 1D Data for stock returns using optuna.
"""

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import optuna
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback
from optuna.pruners import MedianPruner
import sklearn
from sklearn.preprocessing import minmax_scale, robust_scale, scale
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch import cuda, mps
from torch.nn import HuberLoss, L1Loss, MSELoss, ReLU, Mish, Tanh
from torch.optim import Adam, SGD
from pytorch_ranger import Ranger
from ranger21 import Ranger21

from helpers.cross_sectorial import (CS_DATAMODULE_1D)
from models.cross_sectorial import (LSTM, CNN_1D_LSTM)

def objective(trial):
    # Data hyperparameters
    data_type = trial.suggest_categorical("data_type", ["monthly", "daily"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    multistep = trial.suggest_categorical("multistep", [True, False])
    red_each_dim = trial.suggest_categorical("red_each_dim", [None, 1, 2, 4, 6])

    if (red_each_dim is not None):
        red_total_dim = trial.suggest_categorical("red_total_dim", [None, 10, 25, 50, 75, 100])
    else:
        red_total_dim = trial.suggest_categorical("red_total_dim", [10, 25, 50, 75, 100])

    if data_type == "monthly":
        if (not multistep):
            pred_horizon = trial.suggest_categorical("pred_horizon", [1, 2])
        else:
            pred_horizon = 2
        lookback = trial.suggest_categorical("lookback", [1, 3, 6, 12, 24, 36, 48])
    else:
        pred_horizon = trial.suggest_categorical("pred_horizon", [22, 41]) # Data prep stuff requires dis
        lookback = trial.suggest_categorical("lookback", [1, 3, 6, 12, 24, 36, 48])
        lookback = lookback * 21 # Average of 21 trading days per month (252 a year)

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
        loss_fn = MSELoss
    elif loss_name == "huber":
        loss_fn = HuberLoss
    elif loss_name == "l1":
        loss_fn = L1Loss

    activation_name = trial.suggest_categorical("activation", ["relu", "mish", "tanh"])
    if activation_name == "relu":
        activation = ReLU(True)
    elif activation_name == "mish":
        activation = Mish(True)
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
    if multistep and (pred_horizon==1):
        raise optuna.exceptions.TrialPruned("Trial pruned due to invalid combination of multistep and pred_horizon")
    
    if (red_each_dim is not None) and (red_total_dim is not None):
        raise optuna.exceptions.TrialPruned("Trial pruned due to invalid combination of red_each_dim and red_total_dim")

    # Data module
    data = CS_DATAMODULE_1D(
        batch_size=batch_size,
        lookback=lookback,
        pred_horizon=pred_horizon,
        multistep=multistep,
        data_type=data_type,
        train_workers=0,
        overwrite_cache=False,
        pred_target="return",
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
    lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2, 3, 4, 5, 6])
    lstm_nodes = trial.suggest_categorical("lstm_nodes", [32, 64, 128, 256, 512])
    fc_layers = trial.suggest_categorical("fc_layers", [1, 2, 3, 4, 5, 6])
    fc_nodes = trial.suggest_categorical("fc_nodes", [32, 64, 128, 256, 512])

    # Model and local hyperparameters
    model_name = trial.suggest_categorical("model_name", ["LSTM", "CNN_1D_LSTM"])

    if model_name == "LSTM":
        model = LSTM(
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

    else:
        cnn_layers = trial.suggest_categorical("cnn_layers", [1, 2, 3, 4, 5, 6])
        conv_factor = trial.suggest_float("conv_factor", [0.5, 1.5])

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
    #swa = StochasticWeightAveraging(1e-2)


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
        accelerator=DEVICE,
        devices=1,
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        callbacks=[pruning],
    )

    trainer.fit(model, data)

    # Clear GPU cache post-training
    if gpu:
        gpu.empty_cache()

    return trainer.callback_metrics["val_loss"].item()


# Run the Optuna study
if __name__ == "__main__":

    # tensorboard = TensorBoardCallback("optuna_logs/returns", metric_name="val_loss")
    pruner = MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
