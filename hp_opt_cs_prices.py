"""
Script that optimizes hyperparameters for 1D Data for stock returns using optuna.
"""

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import optuna
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback
import sklearn
from sklearn.preprocessing import minmax_scale, robust_scale, scale
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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
    epochs = trial.suggest_categorical("epochs", [500, 1000, 5000, 10_000])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    multistep = trial.suggest_categorical("multistep", [True, False])
    red_each_dim = trial.suggest_categorical("red_each_dim", [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    red_total_dim = trial.suggest_categorical("red_total_dim", [None, 10, 25, 50, 75, 100])

    if data_type == "monthly":
        pred_horizon = trial.suggest_categorical("pred_horizon", [1, 2])
        lookback = trial.suggest_categorical("lookback", 1, 3, 6, 12, 24, 36, 48)
    else:
        pred_horizon = trial.suggest_categorical("pred_horizon", [22, 41]) # Data prep stuff requires dis
        lookback = trial.suggest_categorical("lookback", 1, 3, 6, 12, 24, 36, 48)
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
        loss = MSELoss
    elif loss_name == "huber":
        loss = HuberLoss
    elif loss_name == "l1":
        loss = L1Loss

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

    # Define data module
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

    





    model_name = trial.suggest_categorical("model_name", ["LSTM", "CNN_1D_LSTM"])





    # Model hyperparameters
    cnn_input_size = 1
    lstm_input_size = 64
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 6)
    output_size = 1
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)



    # Prune combinations that are not possible
    if multistep and (pred_horizon==1):
        raise optuna.exceptions.TrialPruned("Trial pruned due to invalid combination of multistep and pred_horizon")
    
    if (red_each_dim is not None) and (red_total_dim is not None):
        if red_each_dim > red_total_dim:
            raise optuna.exceptions.TrialPruned("Trial pruned due to invalid combination of red_each_dim and red_total_dim")




    model = LSTM(n_companies,
            n_features,
            lookback,
            epochs,
            batches_p_epoch,
            lstm_layers=2, 
            lstm_nodes=64, 
            fc_layers=1,
            fc_nodes=64,
            dropout=0, 
            bidirectional=True,
            lr=1e-3, 
            optimizer=Ranger21,
            activation=nn.Mish(True),
            loss_fn=nn.MSELoss())

    trainer = Trainer(
        max_epochs=300,  # Specify the number of epochs you want to train for
        logger=TensorBoardLogger("logs/", name="optuna_logs"),
        callbacks=[EarlyStopping(monitor="val_loss")],
    )

    trainer.fit(model, data_module)
    return trainer.callback_metrics["val_loss"].item()


# Run the Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")

    logger = TensorBoardCallback()
    study.optimize(objective, n_trials=50, callbacks=[wandbc])

    # Access the best parameters
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
