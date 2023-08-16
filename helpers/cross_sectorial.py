"""
Helper functions for cross-sectorial forecasting models
"""
import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TVF
from sklearn.preprocessing import minmax_scale, robust_scale, scale
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from umap import UMAP


class CS_DATAMODULE_1D(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        lookback,
        pred_horizon,
        multistep,
        data_type="monthly",
        train_workers=0,
        overwrite_cache=False,
        pred_target="return",
        cluster=None,
        only_prices=False,
        umap_dim=None,
    ) -> None:
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
        self.only_prices = only_prices
        self.umap_dim = umap_dim

        if data_type == "monthly":
            self.fin_path = "./DATA/Monthly/Processed/month_data_fin_tec.parquet"
            self.macro_path = "./DATA/Monthly/Processed/month_data_macro_USCA.parquet"
            if cluster is not None:
                self.tickers_path = (
                    f"./DATA/Tickers/month_tickers_clean_cluster{self.cluster}.txt"
                )
            else:
                self.tickers_path = "./DATA/Tickers/month_tickers_clean.txt"

        elif data_type == "daily":
            self.fin_path = "./DATA/Daily/Processed/day_data_fin_tec.parquet"
            self.macro_path = "./DATA/Daily/Processed/day_data_macro_USCA.parquet"
            if cluster is not None:
                self.tickers_path = (
                    f"./DATA/Tickers/day_tickers_clean_cluster{self.cluster}.txt"
                )
            else:
                self.tickers_path = "./DATA/Tickers/day_tickers_clean.txt"

        with open(self.tickers_path, "r") as f:
            self.tickers = f.read().strip().split("\n")

    def prepare_data(self):
        if not os.path.exists("./cache/cs/cnn_1D"):
            os.makedirs("./cache/cs/cnn_1D")

        possible_cache_files = [
            f"./cache/cs/cnn_1D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
            f"./cache/cs/cnn_1D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
            f"./cache/cs/cnn_1D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
            f"./cache/cs/cnn_1D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
            f"./cache/cs/cnn_1D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
            f"./cache/cs/cnn_1D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
        ]

        if (all(os.path.exists(file) for file in possible_cache_files)) and (not self.overwrite_cache):
            pass

        else:
            fin = pd.read_parquet(self.fin_path)
            macro = pd.read_parquet(self.macro_path)

            X_train, X_val, X_test, y_train, y_val, y_test = _format_tensors_cs_1D(
                fin_data=fin,
                macro_data=macro,
                tickers=self.tickers,
                lookback=self.lookback,
                pred_horizon=self.pred_horizon,
                multistep=self.multistep,
                onlyprices=self.only_prices,
                pred_target=self.pred_target,
                umap_dim=self.umap_dim,
            )

            np.save(
                f"./cache/cs/cnn_1D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
                X_train,
            )
            np.save(
                f"./cache/cs/cnn_1D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
                X_val,
            )
            np.save(
                f"./cache/cs/cnn_1D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
                X_test,
            )
            np.save(
                f"./cache/cs/cnn_1D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
                y_train,
            )
            np.save(
                f"./cache/cs/cnn_1D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
                y_val,
            )
            np.save(
                f"./cache/cs/cnn_1D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy",
                y_test,
            )

    def setup(self, stage=None):
        self.X_train_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_1D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy"
            )
        ).float()
        self.X_val_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_1D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy"
            )
        ).float()
        self.X_test_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_1D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy"
            )
        ).float()
        self.y_train_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_1D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy"
            )
        ).float()
        self.y_val_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_1D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy"
            )
        ).float()
        self.y_test_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_1D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}_only_prices{self.only_prices}_umap{self.umap_dim}.npy"
            )
        ).float()

        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.val_dataset = TensorDataset(self.X_val_tensor, self.y_val_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )


class CS_DATAMODULE_2D(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        lookback,
        pred_horizon,
        multistep,
        data_type="monthly",
        train_workers=0,
        overwrite_cache=False,
        pred_target="return",
        cluster=None,
    ) -> None:
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
                self.tickers_path = (
                    f"./DATA/Tickers/month_tickers_clean_cluster{self.cluster}.txt"
                )
            else:
                self.tickers_path = "./DATA/Tickers/month_tickers_clean.txt"

        elif data_type == "daily":
            self.fin_path = "./DATA/Daily/Processed/day_data_fin_tec.parquet"
            self.macro_path = "./DATA/Daily/Processed/day_data_macro_USCA.parquet"
            if cluster is not None:
                self.tickers_path = (
                    f"./DATA/Tickers/day_tickers_clean_cluster{self.cluster}.txt"
                )
            else:
                self.tickers_path = "./DATA/Tickers/day_tickers_clean.txt"

        with open(self.tickers_path, "r") as f:
            self.tickers = f.read().strip().split("\n")

    def prepare_data(self):
        if not os.path.exists("./cache/cs/cnn_lstm"):
            os.makedirs("./cache/cs/cnn_lstm")

        possible_cache_files = [
            f"./cache/cs/cnn_lstm/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
            f"./cache/cs/cnn_lstm/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
            f"./cache/cs/cnn_lstm/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
            f"./cache/cs/cnn_lstm/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
            f"./cache/cs/cnn_lstm/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
            f"./cache/cs/cnn_lstm/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
        ]

        if (all(os.path.exists(file) for file in possible_cache_files)) and (not self.overwrite_cache):
            pass

        else:
            fin = pd.read_parquet(self.fin_path)
            macro = pd.read_parquet(self.macro_path)

            X_train, X_val, X_test, y_train, y_val, y_test = _format_tensors_cs_2D(
                fin_data=fin,
                macro_data=macro,
                tickers=self.tickers,
                lookback=self.lookback,
                pred_horizon=self.pred_horizon,
                multistep=self.multistep,
                pred_target=self.pred_target,
            )

            np.save(
                f"./cache/cs/cnn_lstm/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
                X_train,
            )
            np.save(
                f"./cache/cs/cnn_lstm/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
                X_val,
            )
            np.save(
                f"./cache/cs/cnn_lstm/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
                X_test,
            )
            np.save(
                f"./cache/cs/cnn_lstm/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
                y_train,
            )
            np.save(
                f"./cache/cs/cnn_lstm/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
                y_val,
            )
            np.save(
                f"./cache/cs/cnn_lstm/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy",
                y_test,
            )

    def setup(self, stage=None):
        self.X_train_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_lstm/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy"
            )
        ).float()
        self.X_val_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_lstm/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy"
            )
        ).float()
        self.X_test_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_lstm/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy"
            )
        ).float()
        self.y_train_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_lstm/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy"
            )
        ).float()
        self.y_val_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_lstm/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy"
            )
        ).float()
        self.y_test_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/cnn_lstm/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_cluster{self.cluster}.npy"
            )
        ).float()

        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.val_dataset = TensorDataset(self.X_val_tensor, self.y_val_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )


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


def _format_tensors_cs_2D(
    fin_data,
    macro_data,
    tickers,
    lookback=None,
    pred_horizon=1,
    multistep=False,
    pred_target="return",
):
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
        features.loc[~(features[f"{ticker}_CP"] > 0), features.columns] = 0  # Make all rows where the target is 0 also 0
        if pred_target == "return":
            target = (features.filter(regex=f"{ticker}_CP").pct_change(pred_horizon) * 100).iloc[pred_horizon:, :]
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0)
            features = features.iloc[pred_horizon:, :]
        else:
            target = features.filter(regex=f"{ticker}_CP")

        features = features.drop(
            columns=[
                f"{ticker}_CP",
                f"{ticker}_OP",
                f"{ticker}_VOL",
                f"{ticker}_OP",
                f"{ticker}_LP",
                f"{ticker}_HP",
            ]
        )  # Might make model worse but safety against any leakage (appears to actually improve performance)

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
    X_tens = scale(X_tens.reshape(X_tens.shape[0], -1)).reshape(X_tens.shape)  # TODO: make scalers hyperparam
    y_tens = np.stack(ys)

    # Give data shape of (n_samples, channels, timesteps, features)
    X_tens = np.transpose(X_tens, (1, 0, 2, 3))
    if multistep:
        y_tens = np.transpose(y_tens.squeeze(), (1, 0, 2))
    else:
        y_tens = np.transpose(y_tens.squeeze(), (1, 0))

    X_train, X_val, X_test = X_tens[:train_split], X_tens[val_split], X_tens[test_split]
    y_train, y_val, y_test = y_tens[:train_split], y_tens[val_split], y_tens[test_split]

    X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1], X_val.shape[2])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1], X_test.shape[2])

    if multistep:
        y_val = y_val.reshape(1, y_val.shape[0], y_val.shape[1])
        y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])
    else:
        y_val = y_val.reshape(1, y_val.shape[0])
        y_test = y_test.reshape(1, y_test.shape[0])

    return X_train, X_val, X_test, y_train, y_val, y_test


def _format_tensors_cs_1D(
    fin_data,
    macro_data,
    tickers,
    lookback=None,
    pred_horizon=1,
    multistep=False,
    onlyprices=False,
    pred_target="return",
    umap_dim=None,
):
    if not lookback:
        lookback = pred_horizon * 2
    if multistep and (pred_horizon == 1):
        raise ValueError("Multistep only applicable for pred_horizon > 1")
    if onlyprices and (umap_dim is not None):
        print("Skipping UMAP dimensionality reduction since only prices are used")
        umap_dim = None

    X_sequences, y_sequences = [], []

    if onlyprices:
        if pred_target == "return":
            returns = (fin_data.copy().filter(regex="_CP$", axis=1).pct_change(pred_horizon) * 100).iloc[pred_horizon:, :]  # Returns in percentage (since values > 1 needed to turn into pixels)
            features = returns.replace([np.inf, -np.inf], np.nan).fillna(0)  # Missing values set to 0
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
        feature_dfs = []
        target_dfs = []
        for ticker in tqdm(tickers, desc="Preparing Tensors"):
            feature_df = fin_data.copy().filter(regex=f"^{ticker}_")
            feature_df.loc[~(feature_df[f"{ticker}_CP"] > 0), feature_df.columns] = 0  # Make all rows where the target is 0 also 0
            target = feature_df.filter(regex=f"{ticker}_CP")
            target_dfs.append(target.values)
            feature_df = feature_df.drop(
                columns=[
                    f"{ticker}_CP",
                    f"{ticker}_OP",
                    f"{ticker}_VOL",
                    f"{ticker}_OP",
                    f"{ticker}_LP",
                    f"{ticker}_HP",
                ]
            )  # Might make model worse but safety against any leakage (appears to actually improve performance)
            feature_df = scale(feature_df.values)
            if (umap_dim is not None) and (umap_dim <= 100):
                feature_df = UMAP(n_components=umap_dim).fit_transform(feature_df)
            feature_dfs.append(feature_df)
        
        features = np.concatenate(feature_dfs, axis=1)
        # features = _create_timeseries_features(pd.concat([macro_data.copy()] + feature_dfs, axis=1))
        # features = scale(features.values)
        targets = np.concatenate(target_dfs, axis=1)

        # if (umap_dim is not None) and (umap_dim <= 100):
            # features = UMAP(n_components=umap_dim).fit_transform(features) # Change and reduce features of each company? (can add corr removal for each comp?)

        if multistep:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features[i:lookback_idx]
                y_seq = targets[lookback_idx : pred_idx + 1]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        else:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features[i:lookback_idx]
                y_seq = targets[pred_idx]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        X, y = np.array(X_sequences), np.array(y_sequences)

    # Define split indices (since numpy differs from list slicing)
    test_split = -1
    val_split = (pred_horizon + 1) * -1
    train_split = (2 * pred_horizon) * -1

    X_train, X_val, X_test = X[:train_split], X[val_split], X[test_split]
    y_train, y_val, y_test = y[:train_split], y[val_split], y[test_split]

    X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])

    if multistep:
        y_val = y_val.reshape(1, y_val.shape[0], y_val.shape[1])
        y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])
    else:
        y_val = y_val.reshape(1, -1) # Replace with unsqueeze
        y_test = y_test.reshape(1, -1)

    return X_train, X_val, X_test, y_train, y_val, y_test
