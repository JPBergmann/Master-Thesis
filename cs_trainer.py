import os
from typing import Any

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.nn import (HuberLoss, L1Loss, MSELoss, ReLU, Tanh, Mish)
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_ranger import Ranger
from ranger21 import Ranger21
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import minmax_scale, robust_scale, scale
from torch import cuda, mps
from torch.nn import Mish

from helpers.cross_sectorial import (CS_DATAMODULE_1D, CS_DATAMODULE_2D,)
from models.cross_sectorial import (CNN_1D_LSTM, Vanilla_LSTM)

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

    data = CS_DATAMODULE_1D(batch_size=32, 
                            lookback=12, 
                            pred_horizon=1,
                            multistep=False, 
                            data_type="monthly",
                            pred_target="return", 
                            overwrite_cache=False, 
                            scaling_fn=minmax_scale,
                            red_each_dim=1,
                            red_total_dim=None,)

    data.prepare_data()
    data.setup()

    BATCH_SIZE = 32
    N_COMPANIES = int(len(data.tickers))
    N_FEATURES = int(data.X_train_tensor.shape[-1])
    # Batches per epoch which have to be rounded up to the next integer
    N_BATCHES = int(np.ceil(len(data.X_train_tensor) / BATCH_SIZE))
    print(f"Number of batches per Epoch: {N_BATCHES}")

    model = CNN_1D_LSTM(n_companies=N_COMPANIES,
                    n_features=N_FEATURES,
                    lookback=12,
                    epochs=100,
                    batches_p_epoch=N_BATCHES,
                    cnn_layers=6,
                    conv_factor=0.3,
                    lstm_layers=2,
                    lstm_nodes=512,
                    fc_layers=3,
                    fc_nodes=N_COMPANIES,
                    dropout=0.3,
                    bidirectional=True,
                    lr=1e-4,
                    optimizer=Ranger21,
                    activation=Mish(True),
                    loss_fn=HuberLoss(),)


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

    best_model = CNN_1D_LSTM.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path).to(DEVICE)

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
