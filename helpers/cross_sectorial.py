"""
Helper functions for cross-sectorial forecasting models
"""
import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TVF
from sklearn.preprocessing import minmax_scale, scale, robust_scale
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from umap import UMAP


class CS_DATAMODULE(pl.LightningDataModule):
    def __init__(self, batch_size, lookback, pred_horizon, multistep, data_type="monthly", train_workers=0, overwrite_cache=False, pred_target="return", cluster=None) -> None:
        """
        DataModule for the CS baseline model.

        Parameters
        ----------
        batch_size : int
            Batch size for the dataloaders. 
        lookback : int
            Number of months to look back in time for the model.
        pred_horizon : int
            Number of time steps to predict into the future (keep in mind this depends on data_type).
        multistep : bool
            Whether to use multistep prediction or not.
        data_type : {"monthly", "daily"}, default="monthly"
            Whether to use monthly or daily data.
        train_workers : int, default=0
            Number of workers for the train dataloader.        
        """
        super().__init__()
        self.train_workers = train_workers
        self.batch_size = batch_size
        self.lookback = lookback
        self.pred_horizon = pred_horizon
        self.multistep = multistep
        self.data_type = data_type
        self.overwrite_cache = overwrite_cache
        self.pred_target = pred_target
        self.cluster = cluster

        if data_type == "monthly":
            self.fin_path = "./DATA/Monthly/Processed/month_data_fin_tec.parquet"
            self.macro_path = "./DATA/Monthly/Processed/month_data_macro_USCA.parquet"
            if cluster is not None:
                self.tickers_path = f"./DATA/Tickers/month_tickers_clean_cluster{self.cluster}.txt"
            else:
                self.tickers_path = "./DATA/Tickers/month_tickers_clean.txt"

        elif data_type == "daily":
            self.fin_path = "./DATA/Daily/Processed/day_data_fin_tec.parquet"
            self.macro_path = "./DATA/Daily/Processed/day_data_macro_USCA.parquet"
            if cluster is not None:
                self.tickers_path = f"./DATA/Tickers/day_tickers_clean_cluster{self.cluster}.txt"
            else:
                self.tickers_path = "./DATA/Tickers/day_tickers_clean.txt"

        with open(self.tickers_path, "r") as f:
            self.tickers = f.read().strip().split("\n")

    def prepare_data(self):

        if not os.path.exists("./cache/cs/cnn_lstm"):
            os.makedirs("./cache/cs/cnn_lstm")

        possible_cache_files = [
            f"./cache/cs/cnn_lstm/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/cnn_lstm/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/cnn_lstm/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/cnn_lstm/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/cnn_lstm/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/cnn_lstm/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy"
        ]

        if (all(os.path.exists(file) for file in possible_cache_files)) and (not self.overwrite_cache):
            pass

        else:
        
            fin = pd.read_parquet(self.fin_path)
            macro = pd.read_parquet(self.macro_path)

            X_train, X_val, X_test, y_train, y_val, y_test = _format_tensors_cs(fin_data=fin, 
                                                                            macro_data=macro,
                                                                            tickers=self.tickers,
                                                                            lookback=self.lookback, 
                                                                            pred_horizon=self.pred_horizon,
                                                                            multistep=self.multistep,
                                                                            pred_target=self.pred_target)

            np.save(f"./cache/cs/cnn_lstm/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", X_train)
            np.save(f"./cache/cs/cnn_lstm/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", X_val)
            np.save(f"./cache/cs/cnn_lstm/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", X_test)
            np.save(f"./cache/cs/cnn_lstm/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", y_train)
            np.save(f"./cache/cs/cnn_lstm/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", y_val)
            np.save(f"./cache/cs/cnn_lstm/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", y_test)

    def setup(self, stage=None):
        self.X_train_tensor = torch.from_numpy(np.load(f"./cache/cs/cnn_lstm/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.X_val_tensor = torch.from_numpy(np.load(f"./cache/cs/cnn_lstm/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.X_test_tensor = torch.from_numpy(np.load(f"./cache/cs/cnn_lstm/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.y_train_tensor = torch.from_numpy(np.load(f"./cache/cs/cnn_lstm/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.y_val_tensor = torch.from_numpy(np.load(f"./cache/cs/cnn_lstm/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.y_test_tensor = torch.from_numpy(np.load(f"./cache/cs/cnn_lstm/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()

        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.val_dataset = TensorDataset(self.X_val_tensor, self.y_val_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)


class CS_VID_DATAMODULE(pl.LightningDataModule):
    def __init__(self, batch_size, lookback, pred_horizon, multistep, data_type, train_workers=0, resize=None, overwrite_cache=False, pred_target="return") -> None:
        """
        DataModule for the CS baseline model.

        Parameters
        ----------
        batch_size : int
            Batch size for the dataloaders. 
        lookback : int
            Number of months to look back in time for the model.
        pred_horizon : int
            Number of time steps to predict into the future (keep in mind this depends on data_type).
        multistep : bool
            Whether to use multistep prediction or not.
        data_type : {"monthly", "daily"}, default="monthly"
            Whether to use monthly or daily data.
        train_workers : int, default=0
            Number of workers for the train dataloader.   
        resize : tuple, default=None
            If not None, resize the images to the given size.     
        """
        super().__init__()
        self.train_workers = train_workers
        self.batch_size = batch_size
        self.lookback = lookback
        self.pred_horizon = pred_horizon
        self.multistep = multistep
        self.data_type = data_type
        self.resize = resize
        self.overwrite_cache = overwrite_cache
        self.pred_target = pred_target

        if data_type == "monthly":
            self.fin_path = "./DATA/Monthly/Processed/month_data_fin_tec.parquet"
            self.macro_path = "./DATA/Monthly/Processed/month_data_macro_USCA.parquet"
            self.tickers_path = "./DATA/Tickers/month_tickers_clean.txt"

        elif data_type == "daily":
            self.fin_path = "./DATA/Daily/Processed/day_data_fin_tec.parquet"
            self.macro_path = "./DATA/Daily/Processed/day_data_macro_USCA.parquet"
            self.tickers_path = "./DATA/Tickers/day_tickers_clean.txt"

    def prepare_data(self):

        if not os.path.exists("./cache/cs/convlstm_ae"):
            os.makedirs("./cache/cs/convlstm_ae")

        possible_cache_files = [
            f"./cache/cs/convlstm_ae/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/convlstm_ae/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/convlstm_ae/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/convlstm_ae/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/convlstm_ae/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy",
            f"./cache/cs/convlstm_ae/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy"
        ]

        if (all(os.path.exists(file) for file in possible_cache_files)) and (not self.overwrite_cache):
            pass

        else:
        
            fin = pd.read_parquet(self.fin_path)
            # macro = pd.read_parquet(self.macro_path)
            # with open(self.tickers_path, "r") as f:
            #     tickers = f.read().strip().split("\n")

            X_train, X_val, X_test, y_train, y_val, y_test = _format_tensors_cs_vid(fin_data=fin,
                                                                                    lookback=self.lookback, 
                                                                                    pred_horizon=self.pred_horizon,
                                                                                    multistep=self.multistep,
                                                                                    resize=self.resize,
                                                                                    pred_target=self.pred_target)

            np.save(f"./cache/cs/convlstm_ae/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", X_train)
            np.save(f"./cache/cs/convlstm_ae/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", X_val)
            np.save(f"./cache/cs/convlstm_ae/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", X_test)
            np.save(f"./cache/cs/convlstm_ae/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", y_train)
            np.save(f"./cache/cs/convlstm_ae/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", y_val)
            np.save(f"./cache/cs/convlstm_ae/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy", y_test)

    def setup(self, stage=None):
        self.X_train_tensor = torch.from_numpy(np.load(f"./cache/cs/convlstm_ae/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.X_val_tensor = torch.from_numpy(np.load(f"./cache/cs/convlstm_ae/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.X_test_tensor = torch.from_numpy(np.load(f"./cache/cs/convlstm_ae/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.y_train_tensor = torch.from_numpy(np.load(f"./cache/cs/convlstm_ae/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.y_val_tensor = torch.from_numpy(np.load(f"./cache/cs/convlstm_ae/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()
        self.y_test_tensor = torch.from_numpy(np.load(f"./cache/cs/convlstm_ae/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}.npy")).float()

        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.val_dataset = TensorDataset(self.X_val_tensor, self.y_val_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    
def _create_timeseries_features(dataframe):
    df = dataframe.copy()

    # Date Features
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["day_of_year"] = df.index.dayofyear
    df["week_of_year"] = df.index.isocalendar().week
    df["month_of_year"] = df.index.month
    df["quarter_of_year"] = df.index.quarter
    df["year"] = df.index.year

    return df


def _format_tensors_cs_vid(fin_data, lookback=None, pred_horizon=1, multistep=False, resize=None, pred_target="return"):
    if not lookback:
        lookback = pred_horizon * 2
    if multistep and pred_horizon == 1:
        raise ValueError("Multistep only applicable for pred_horizon > 1")

    if pred_target == "return":
        returns = (fin_data.copy().filter(regex="_CP$", axis=1).pct_change(pred_horizon) * 100).iloc[pred_horizon:, :9] # Returns in percentage (since values > 1 needed to turn into pixels)
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0) # Missing values set to 0

    else:
        returns = fin_data.copy().filter(regex="_CP$", axis=1).iloc[:, :400] # Actual prices
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0) # Missing values set to 0

    # Turn each timestep into a picture (20x20 pixels) of returns (0-255) - 0 is black, 255 is white - 0 is lowest return, 255 is highest return
    features = []
    for i in tqdm(range(len(returns)), desc="Converting returns into video sequences"):
        timestep = returns.iloc[i, :].values # Scale each img seperately?
        img = (torch.from_numpy(timestep).float().sigmoid() * 255).reshape(3, 3)
        if resize:
            img = TVF.resize(img.unsqueeze(dim=0), resize, antialias=True).squeeze()

        features.append(img/255) # scale to 0-1

    features = torch.stack(features)
    X_sequences, y_sequences = [], []

    if multistep:
        for i in tqdm(range(len(features) - lookback), desc="Formatting tensors"):
            lookback_idx = i + lookback
            pred_idx = lookback_idx + pred_horizon - 1

            if pred_idx > len(features) - 1:
                continue

            X_seq = features[i:lookback_idx]
            y_seq = features[lookback_idx : pred_idx + 1]
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

    else:
        for i in tqdm(range(len(features) - lookback), desc="Formatting tensors"):
            lookback_idx = i + lookback
            pred_idx = lookback_idx + pred_horizon - 1

            if pred_idx > len(features) - 1:
                continue

            X_seq = features[i:lookback_idx]
            y_seq = features[pred_idx]
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

    X, y = torch.stack(X_sequences), torch.stack(y_sequences)
    
    X = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2], X.shape[3]) # Add channel dimension (1 since grayscale)
    
    if multistep:
        y = y.reshape(y.shape[0], y.shape[1], 1, y.shape[2], y.shape[3]) # Add channel dimension (1 since grayscale)
    else:
        y = y.reshape(y.shape[0], 1, y.shape[1], y.shape[2]) # Add channel dimension (1 since grayscale)

    # Define split indices (since numpy differs from list slicing)
    test_split = -1
    val_split = (pred_horizon + 1) * -1
    train_split = (2 * pred_horizon) * -1

    # train_split_idx = int(len(X) * 0.8)
    # val_split_idx = int(len(X) * 0.95)

    # X_train, X_val, X_test = X[:train_split_idx], X[train_split_idx:val_split_idx], X[val_split_idx:]
    # y_train, y_val, y_test = y[:train_split_idx], y[train_split_idx:val_split_idx], y[val_split_idx:]

    X_train, X_val, X_test = X[:train_split], X[val_split], X[test_split]
    y_train, y_val, y_test = y[:train_split], y[val_split], y[test_split]
    
    X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1], X_val.shape[2], X_val.shape[3])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1], X_test.shape[2], X_val.shape[3])
    
    if multistep:
        y_val = y_val.reshape(1, y_val.shape[0], y_val.shape[1], y_val.shape[2],  y_val.shape[3])
        y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1], y_test.shape[2],  y_val.shape[3])
    else:
        y_val = y_val.reshape(1, y_val.shape[0], y_val.shape[1], y_val.shape[2])
        y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1], y_test.shape[2])

    return X_train, X_val, X_test, y_train, y_val, y_test


def _format_tensors_cs(fin_data, macro_data, tickers, lookback=None, pred_horizon=1, multistep=False, pred_target="return"):
    if not lookback:
        lookback = pred_horizon * 2
    if multistep and pred_horizon == 1:
        raise ValueError("Multistep only applicable for pred_horizon > 1")

    Xs = []
    ys = []

    for ticker in tqdm(tickers, desc="Preparing Tensors"):
        fin_df = _create_timeseries_features(fin_data.copy().filter(regex=f"^{ticker}_"))
        macro_df = macro_data.copy()

        features = pd.concat([fin_df, macro_df], axis=1)
        features.loc[~(features[f"{ticker}_CP"] > 0), features.columns] = 0 # Make all rows where the target is 0 also 0
        if pred_target == "return":
            target = (features.filter(regex=f"{ticker}_CP").pct_change(pred_horizon) * 100).iloc[pred_horizon:, :]
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0)
            features = features.iloc[pred_horizon:, :]
        else:
            target = features.filter(regex=f"{ticker}_CP")
        # features = features.drop(
        #     columns=[
        #         f"{ticker}_CP",
        #         f"{ticker}_OP",
        #         f"{ticker}_VOL",
        #         f"{ticker}_OP",
        #         f"{ticker}_LP",
        #         f"{ticker}_HP",
        #     ]
        # )  # Might make model worse but safety against any leakage (appears to actually improve performance)

        X_sequences, y_sequences = [], []

        if multistep:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features.iloc[i:lookback_idx]
                y_seq = target.iloc[lookback_idx : pred_idx + 1]
                X_sequences.append(X_seq.values)
                y_sequences.append(y_seq.values)

        else:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features.iloc[i:lookback_idx]
                y_seq = target.iloc[pred_idx]
                X_sequences.append(X_seq.values)
                y_sequences.append(y_seq.values)

        X, y = np.array(X_sequences), np.array(y_sequences)

        Xs.append(X)
        ys.append(y)

    # Define split indices (since numpy differs from list slicing)
    test_split = -1
    val_split = (pred_horizon + 1) * -1
    train_split = (2 * pred_horizon) * -1

    X_tens = np.stack(Xs)
    X_tens = scale(X_tens.reshape(X_tens.shape[0], -1)).reshape(X_tens.shape) # TODO: make scalers hyperparam
    y_tens = np.stack(ys)

    # Give data shape of (n_samples, channels, timesteps, features)
    X_tens = np.transpose(X_tens, (1, 0, 2, 3))
    y_tens = np.transpose(y_tens.squeeze(), (1, 0))
    
    #return X_tens, y_tens
    X_train, X_val, X_test = X_tens[:train_split], X_tens[val_split], X_tens[test_split]
    y_train, y_val, y_test = y_tens[:train_split], y_tens[val_split], y_tens[test_split]

    # Walk forward validation (validation is always the trained data + 1) ??????????????

    X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1], X_val.shape[2])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1], X_test.shape[2])

    y_val = y_val.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    return X_train, X_val, X_test, y_train, y_val, y_test


def _format_tensors_cs_1dim(fin_data, macro_data, tickers, lookback=None, pred_horizon=1, multistep=False, onlyprices=False, pred_target="return", umap_dim=None):
    if not lookback:
        lookback = pred_horizon * 2
    if multistep and pred_horizon == 1:
        raise ValueError("Multistep only applicable for pred_horizon > 1")

    X_sequences, y_sequences = [], []
        
    if onlyprices:

        if pred_target == "return":
            returns = (fin_data.copy().filter(regex="_CP$", axis=1).pct_change(pred_horizon) * 100).iloc[pred_horizon:, :] # Returns in percentage (since values > 1 needed to turn into pixels)
            features = returns.replace([np.inf, -np.inf], np.nan).fillna(0) # Missing values set to 0
        else:
            features = fin_data.copy().filter(regex="_CP$", axis=1)

        features = pd.DataFrame(scale(features.values), columns=features.columns, index=features.index)
        if multistep:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features.iloc[i:lookback_idx]
                y_seq = features.iloc[lookback_idx : pred_idx + 1]
                X_sequences.append(X_seq.values)
                y_sequences.append(y_seq.values)

        else:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features.iloc[i:lookback_idx]
                y_seq = features.iloc[pred_idx]
                X_sequences.append(X_seq.values)
                y_sequences.append(y_seq.values)

    else:
        """
        - ADD MULTICOLINEARITY THRESHOLD
        - List of DFs for each company concat to feature df. then add timeseries and also concat macro.
        """
        feature_dfs = []
        targets = pd.DataFrame()
        for ticker in tqdm(tickers, desc="Preparing Tensors"):
            feature_df = fin_data.copy().filter(regex=f"^{ticker}_")
            feature_df.loc[~(feature_df[f"{ticker}_CP"] > 0), feature_df.columns] = 0 # Make all rows where the target is 0 also 0
            target = feature_df.filter(regex=f"{ticker}_CP")
            targets[f"y_{ticker}"] = target
            feature_df = feature_df.drop(
                columns=[
                    f"{ticker}_CP",
                    f"{ticker}_OP",
                    f"{ticker}_VOL",
                    f"{ticker}_OP",
                    f"{ticker}_LP",
                    f"{ticker}_HP",
                ]
            ) # Might make model worse but safety against any leakage (appears to actually improve performance)
            feature_dfs.append(feature_df)
        
        features = _create_timeseries_features(pd.concat([macro_data.copy()] + feature_dfs, axis=1))
        features = pd.DataFrame(scale(features.values), columns=features.columns, index=features.index)

        if umap_dim:
            features = UMAP(n_components=umap_dim).fit_transform(features)

        if multistep:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features.iloc[i:lookback_idx]
                y_seq = target.iloc[lookback_idx : pred_idx + 1]
                X_sequences.append(X_seq.values)
                y_sequences.append(y_seq.values)

        else:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features.iloc[i:lookback_idx]
                y_seq = target.iloc[pred_idx]
                X_sequences.append(X_seq.values)
                y_sequences.append(y_seq.values)

        X, y = np.array(X_sequences), np.array(y_sequences)

    # Define split indices (since numpy differs from list slicing)
    test_split = -1
    val_split = (pred_horizon + 1) * -1
    train_split = (2 * pred_horizon) * -1

    X_train, X_val, X_test = X[:train_split], X[val_split], X[test_split]
    y_train, y_val, y_test = y[:train_split], y[val_split], y[test_split]

    X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])

    return X_train, X_val, X_test, y_train, y_val, y_test