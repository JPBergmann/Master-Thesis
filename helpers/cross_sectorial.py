"""
Helper functions for cross-sectorial forecasting models
"""
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TVF
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import minmax_scale, robust_scale, scale, label_binarize
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
        red_each_dim=None,
        red_total_dim=None,
        scaling_fn=scale
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
        self.red_each_dim = red_each_dim
        self.red_total_dim = red_total_dim
        self.scaling_fn = scaling_fn

        if data_type == "monthly":
            self.fin_path = "./DATA/Monthly/Processed/month_data_fin_tec.parquet"
            self.macro_path = "./DATA/Monthly/Processed/month_data_macro_USCA.parquet"
            self.tickers_path = "./DATA/Tickers/month_tickers_clean.txt"

        elif data_type == "daily":
            self.fin_path = "./DATA/Daily/Processed/day_data_fin_tec.parquet"
            self.macro_path = "./DATA/Daily/Processed/day_data_macro_USCA.parquet"
            self.tickers_path = "./DATA/Tickers/day_tickers_clean.txt"

        with open(self.tickers_path, "r") as f:
            self.tickers = f.read().strip().split("\n")

    def prepare_data(self):
        if not os.path.exists("./cache/cs/1D"):
            os.makedirs("./cache/cs/1D")

        possible_cache_files = [
            f"./cache/cs/1D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
            f"./cache/cs/1D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
            f"./cache/cs/1D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
            f"./cache/cs/1D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
            f"./cache/cs/1D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
            f"./cache/cs/1D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
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
                pred_target=self.pred_target,
                red_each_dim=self.red_each_dim,
                red_total_dim=self.red_total_dim,
                scaling_fn=self.scaling_fn,
            )

            np.save(
                f"./cache/cs/1D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
                X_train,
            )
            np.save(
                f"./cache/cs/1D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
                X_val,
            )
            np.save(
                f"./cache/cs/1D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
                X_test,
            )
            np.save(
                f"./cache/cs/1D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
                y_train,
            )
            np.save(
                f"./cache/cs/1D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
                y_val,
            )
            np.save(
                f"./cache/cs/1D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy",
                y_test,
            )

    def setup(self, stage=None):
        self.X_train_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/1D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy"
            )
        ).float()
        self.X_val_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/1D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy"
            )
        ).float()
        self.X_test_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/1D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy"
            )
        ).float()
        self.y_train_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/1D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy"
            )
        ).float()
        self.y_val_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/1D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy"
            )
        ).float()
        self.y_test_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/1D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_red_each_dim{self.red_each_dim}_red_total_dim{self.red_total_dim}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}.npy"
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
        tickers,
        data_type="monthly",
        train_workers=0,
        overwrite_cache=False,
        pred_target="return",
        scaling_fn=scale,
        cluster=None,
        goal="classification",
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
        self.tickers = tickers
        self.data_type = data_type
        self.overwrite_cache = overwrite_cache
        self.pred_target = pred_target
        self.scaling_fn = scaling_fn
        self.cluster = cluster
        self.goal = goal

        if data_type == "monthly":
            self.fin_path = "./DATA/Monthly/Processed/month_data_fin_tec.parquet"
            self.macro_path = "./DATA/Monthly/Processed/month_data_macro_USCA.parquet"
            if (cluster is not None):
                self.tickers_path = f"./DATA/Tickers/month_tickers_clean_cluster{cluster}.txt"
            else:
                self.tickers_path = "./DATA/Tickers/month_tickers_clean.txt"

        elif data_type == "daily":
            self.fin_path = "./DATA/Daily/Processed/day_data_fin_tec.parquet"
            self.macro_path = "./DATA/Daily/Processed/day_data_macro_USCA.parquet"
            if (cluster is not None):
                self.tickers_path = f"./DATA/Tickers/day_tickers_clean_cluster{cluster}.txt"
            else:
                self.tickers_path = "./DATA/Tickers/day_tickers_clean.txt"

        #with open(self.tickers_path, "r") as f:
            #self.tickers = f.read().strip().split("\n")

    def prepare_data(self):
        if not os.path.exists("./cache/cs/2D"):
            os.makedirs("./cache/cs/2D")

        possible_cache_files = [
            f"./cache/cs/2D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
            f"./cache/cs/2D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
            f"./cache/cs/2D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
            f"./cache/cs/2D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
            f"./cache/cs/2D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
            f"./cache/cs/2D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
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
                scaling_fn=self.scaling_fn,
                goal=self.goal
            )

            np.save(
                f"./cache/cs/2D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
                X_train,
            )
            np.save(
                f"./cache/cs/2D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
                X_val,
            )
            np.save(
                f"./cache/cs/2D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
                X_test,
            )
            np.save(
                f"./cache/cs/2D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
                y_train,
            )
            np.save(
                f"./cache/cs/2D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
                y_val,
            )
            np.save(
                f"./cache/cs/2D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy",
                y_test,
            )

    def setup(self, stage=None):
        self.X_train_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/2D/X_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy"
            )
        ).float()
        self.X_val_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/2D/X_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy"
            )
        ).float()
        self.X_test_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/2D/X_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy"
            )
        ).float()
        self.y_train_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/2D/y_train_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy"
            )
        ).float()
        self.y_val_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/2D/y_val_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy"
            )
        ).float()
        self.y_test_tensor = torch.from_numpy(
            np.load(
                f"./cache/cs/2D/y_test_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_scaler{self.scaling_fn.__name__}_pred_target{self.pred_target}_cluster{self.cluster}.npy"
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


def _format_tensors_cs_1D(
    fin_data,
    macro_data,
    tickers,
    lookback=None,
    pred_horizon=1,
    multistep=False,
    pred_target="return",
    red_each_dim=None,
    red_total_dim=None,
    scaling_fn=scale
):
    if not lookback:
        lookback = pred_horizon * 2
    if multistep and (pred_horizon == 1):
        raise ValueError("Multistep only applicable for pred_horizon > 1")
    if (red_each_dim is not None) and (red_total_dim is not None):
        raise ValueError("Cannot stack more than 1 dimensionality reduction method")
    if (red_total_dim is not None) and (red_total_dim > 100):
        raise ValueError("Cannot reduce to more than 100 dimensions")

    X_sequences, y_sequences = [], []

    feature_dfs = []
    target_dfs = []

    for ticker in tqdm(tickers, desc="Preparing Tensors"):
        feature_df = fin_data.copy().filter(regex=f"^{ticker}_")
        feature_df.loc[~(feature_df[f"{ticker}_CP"] > 0), feature_df.columns] = 0  # Make all rows where the target is 0 also 0
    
        if pred_target == "return":
            target = (feature_df.filter(regex=f"{ticker}_CP").pct_change(pred_horizon) * 100).iloc[pred_horizon:, :]  # Returns in percentage to resolve numerical conflicts
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0)  # Missing values set to 0 if any
            feature_df = feature_df.iloc[pred_horizon:, :]
        else:
            target = feature_df.filter(regex=f"{ticker}_CP")

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
        
        if (red_each_dim is not None):
            if pred_target == "return":
                if scaling_fn == minmax_scale:
                    feature_df = scaling_fn(np.concatenate([_create_timeseries_features(macro_data.copy()).iloc[pred_horizon:, :].values, feature_df], axis=1), feature_range=(-1, 1))
                else:
                    feature_df = scaling_fn(np.concatenate([_create_timeseries_features(macro_data.copy()).iloc[pred_horizon:, :].values, feature_df], axis=1))
            else:
                if scaling_fn == minmax_scale:
                    feature_df = scaling_fn(np.concatenate([_create_timeseries_features(macro_data.copy()).values, feature_df], axis=1), feature_range=(0, 1))
                else:
                    feature_df = scaling_fn(np.concatenate([_create_timeseries_features(macro_data.copy()).values, feature_df], axis=1))

            kpca = KernelPCA(n_components=red_each_dim)
            feature_df = kpca.fit_transform(feature_df)

        feature_dfs.append(feature_df)
        target_dfs.append(target.values)

    if pred_target == "return":
        features = np.concatenate([_create_timeseries_features(macro_data.copy()).iloc[pred_horizon:, :].values] + feature_dfs, axis=1)
        features = scaling_fn(features, feature_range=(-1, 1)) if (scaling_fn == minmax_scale) else scaling_fn(features)
    else:
        features = np.concatenate([_create_timeseries_features(macro_data.copy()).values] + feature_dfs, axis=1)
        features = scaling_fn(features, feature_range=(0, 1)) if (scaling_fn == minmax_scale) else scaling_fn(features)

    targets = np.concatenate(target_dfs, axis=1)

    if (red_total_dim is not None):
        features = UMAP(n_components=red_total_dim).fit_transform(features) # Change and reduce features of each company? (can add corr removal for each comp?)

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

    X, y = np.array(X_sequences, dtype=np.float32), np.array(y_sequences, dtype=np.float32)

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
        y_val = y_val.reshape(1, -1)
        y_test = y_test.reshape(1, -1)

    return X_train, X_val, X_test, y_train, y_val, y_test


def _format_tensors_cs_2D(
    fin_data,
    macro_data,
    tickers,
    lookback=None,
    pred_horizon=1,
    multistep=False,
    pred_target="return",
    scaling_fn=scale,
    goal="classification",
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
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0).values
            if goal == "classification":
                bins = [-np.inf, -15, -5, 5, 15, np.inf]
                labels = [0, 1, 2, 3, 4]
                target = pd.cut(target.squeeze(), bins=bins, labels=labels, include_lowest=True).to_numpy()
                target = label_binarize(target, classes=labels)
        else:
            target = features.filter(regex=f"{ticker}_CP").values

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
        features = features.astype(np.float64)
        # features = np.log1p(features).replace([np.inf, -np.inf], np.nan).fillna(0)

        if pred_target == "return":
            features = np.log1p(features.diff(pred_horizon)).iloc[pred_horizon:, :]
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)

            if scaling_fn == minmax_scale:
                features = scaling_fn(features, feature_range=(-1, 1))
            else:
                features = scaling_fn(features)
        else:
            if scaling_fn == minmax_scale:
                features = scaling_fn(features, feature_range=(0, 1))
            else:
                features = scaling_fn(features)

        X_sequences, y_sequences = [], []

        if multistep:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features[i:lookback_idx]
                y_seq = target[lookback_idx : pred_idx + 1]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        else:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features[i:lookback_idx]
                y_seq = target[pred_idx]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        X, y = np.array(X_sequences), np.array(y_sequences)

        Xs.append(X)
        ys.append(y)

    # Define split indices (since numpy differs from list slicing)
    test_split = -1
    val_split = -2

    X_tens = np.stack(Xs, dtype=np.float32)
    #if scaling_fn == minmax_scale:
        #X_tens = scaling_fn(X_tens.reshape(X_tens.shape[0], -1), feature_range=(-1, 1)).reshape(X_tens.shape)
    #else:
        #X_tens = scaling_fn(X_tens.reshape(X_tens.shape[0], -1)).reshape(X_tens.shape)
    y_tens = np.stack(ys, dtype=np.float32)

    # Give data shape of (n_samples, channels, timesteps, features)
    X_tens = np.transpose(X_tens, (1, 0, 2, 3))

    if multistep or (goal == "classification"):
        y_tens = np.transpose(y_tens.squeeze(), (1, 0, 2))
    else:
        y_tens = np.transpose(y_tens.squeeze(), (1, 0))

    X_train, X_val, X_test = X_tens[:val_split], X_tens[val_split], X_tens[test_split]
    y_train, y_val, y_test = y_tens[:val_split], y_tens[val_split], y_tens[test_split]

    X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1], X_val.shape[2])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1], X_test.shape[2])

    if multistep or (goal == "classification"):
        y_val = y_val.reshape(1, y_val.shape[0], y_val.shape[1])
        y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])
    else:
        y_val = y_val.reshape(1, y_val.shape[0])
        y_test = y_test.reshape(1, y_test.shape[0])

    return X_train, X_val, X_test, y_train, y_val, y_test
