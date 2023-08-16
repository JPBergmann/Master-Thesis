import lightning.pytorch as pl
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback
import pandas as pd
import sklearn
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import cuda, mps
from torch.nn import HuberLoss, L1Loss, MSELoss, ReLU, Mish, Tanh
from torch.optim import Adam, SGD
from pytorch_ranger import Ranger
from ranger21 import Ranger21

from helpers.cross_sectorial import (CS_DATAMODULE_1D, CS_DATAMODULE_2D,
                                     CS_VID_DATAMODULE)
from models.cross_sectorial import (CNN_1D_LSTM, CNN_2D_LSTM, MH_CNN_1D_LSTM,
                                    ConvLSTM_AE)

def objective(trial):
    # Model hyperparameters
    cnn_input_size = 1
    lstm_input_size = 64
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 6)
    output_size = 1
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)

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

    # Datamodule hyperparameters
    lookback = trial.suggest_int("lookback", 3, 48)
    multistep = trial.suggest_categorical("multistep", [True, False])

    if data_type == "monthly": # Prob Seperate file for daily vs monthly and cs vs iterative
        pred_horizon = trial.suggest_categorical("pred_horizon", [1, 2])
    else:
        pred_horizon = trial.suggest_categorical("pred_horizon", [19, 42]) # Since validation month (April) is 20 trading days

    # Trainer hyperparameters
    data_type = trial.suggest_categorical("data_type", ["monthly", "daily"])
    epochs = trial.suggest_categorical("epochs", [100, 500, 1000, 5000, 10_000])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    # Prune combinations that are not possible
    if multistep and (pred_horizon==1):
        raise optuna.exceptions.TrialPruned("Trial pruned due to invalid combination of multistep and pred_horizon")


    model = CNN_1D_LSTM(
        cnn_input_size=cnn_input_size,
        lstm_input_size=lstm_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        lookback=lookback,
        dropout=dropout,
        lr=lr,
        optimizer=optimizer,
        loss=loss
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
        max_epochs=300,  # Specify the number of epochs you want to train for
        logger=TensorBoardLogger("logs/", name="optuna_logs"),
        callbacks=[EarlyStopping(monitor="val_loss")],
    )

    trainer.fit(model, data_module)
    return trainer.callback_metrics["val_loss"].item()


# Run the Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")

    wandbc = WeightsAndBiasesCallback()
    study.optimize(objective, n_trials=50, callbacks=[wandbc])

    # Access the best parameters
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
