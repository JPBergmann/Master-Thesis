import lightning.pytorch as pl
import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_ranger import Ranger
from ranger21 import Ranger21
from torch import cuda, mps
from torch.nn import HuberLoss, L1Loss, MSELoss, optim

from helpers.cross_sectorial import (CS_DATAMODULE_1D, CS_DATAMODULE_2D,
                                     CS_VID_DATAMODULE)
from models.cross_sectorial import (CNN_1D_LSTM, CNN_2D_LSTM, MH_CNN_1D_LSTM,
                                    ConvLSTM_AE)


# Example hyperparameters for the CNN_1D_LSTM model
def objective(trial):
    cnn_input_size = 1
    lstm_input_size = 64
    hidden_size = trial.suggest_int("hidden_size", 64, 512)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    output_size = 1
    lookback = trial.suggest_int("lookback", 3, 48)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    loss_name = trial.suggest_categorical("loss", ["mse", "huber", "l1"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    data_type = trial.suggest_categorical("data_type", ["monthly", "daily"])
    if data_type == "monthly":
        pred_horizon = trial.suggest_categorical("pred_horizon", [1, 2])
    else:
        pred_horizon = trial.suggest_categorical("pred_horizon", [19, 42]) # Since validation month (April) is 20 trading days

    # Prune combinations that are not possible
    if multistep and (pred_horizon==1):
        optuna.TrialPruned()
    optimizer_name = trial.suggest_categorical("optimizer", ["Ranger21", "Ranger", "Adam", "SGD"])

    if optimizer_name == "Ranger21":
        optimizer_cls = Ranger21
    elif optimizer_name == "Ranger":
        optimizer_cls = Ranger
    elif optimizer_name == "Adam":
        optimizer_cls = optim.Adam
    elif optimizer_name == "SGD":
        optimizer_cls = optim.SGD

    if loss_name == "mse":
        loss_cls = MSELoss
    elif loss_name == "huber":
        loss_cls = HuberLoss
    elif loss_name == "l1":
        loss_cls = L1Loss

    model = CNN_1D_LSTM(
        cnn_input_size=cnn_input_size,
        lstm_input_size=lstm_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        lookback=lookback,
        dropout=dropout,
        lr=lr,
        optimizer=optimizer_cls,
        loss=loss_cls
    )

    data_module = CS_DATAMODULE_1D(
        batch_size=batch_size,
        lookback=lookback,
        pred_horizon=1,
        multistep=False,
        data_type="monthly",
        train_workers=4,
        pred_target="return",
        cluster=None,
        only_prices=False,
        umap_dim=None,
    )

    trainer = Trainer(
        max_epochs=...,  # Specify the number of epochs you want to train for
        logger=TensorBoardLogger("logs/", name="optuna_logs"),
        callbacks=[EarlyStopping(monitor="val_loss")],
    )

    trainer.fit(model, data_module)
    return trainer.callback_metrics["val_loss"].item()


# Run the Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # You can adjust the number of trials

    # Access the best parameters
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
