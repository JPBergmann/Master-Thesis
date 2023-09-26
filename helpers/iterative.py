"""
Helper functions for iterative forecasting models
"""
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TVF
from sklearn.preprocessing import minmax_scale, robust_scale, scale
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class IT_DATAMODULE(pl.LightningDataModule):
    def __init__(self, batch_size, lookback, pred_horizon, multistep, ticker_idx, data_type="monthly", train_workers=0, overwrite_cache=False, pred_target="return", multicolinearity_threshold=None) -> None:
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
        self.ticker_idx = ticker_idx
        self.pred_target = pred_target
        self.multicolinearity_threshold = multicolinearity_threshold

        if data_type == "monthly":
            self.fin_path = "./DATA/Monthly/Processed/month_data_fin_tec.parquet"
            self.macro_path = "./DATA/Monthly/Processed/month_data_macro_USCA.parquet"
            self.tickers_path = "./DATA/Tickers/month_tickers_clean.txt"

        elif data_type == "daily":
            self.fin_path = "./DATA/Daily/Processed/day_data_fin_tec.parquet"
            self.macro_path = "./DATA/Daily/Processed/day_data_macro_USCA.parquet"
            self.tickers_path = "./DATA/Tickers/day_tickers_clean.txt"

    def prepare_data(self):

        if not os.path.exists("./cache/it"):
            os.makedirs("./cache/it")

        possible_cache_files = [
            f"./cache/it/X_train_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy",
            f"./cache/it/X_val_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy",
            f"./cache/it/X_test_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy",
            f"./cache/it/y_train_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy",
            f"./cache/it/y_val_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy",
            f"./cache/it/y_test_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy"
        ]

        if (all(os.path.exists(file) for file in possible_cache_files)) and (not self.overwrite_cache):
            pass

        else:
        
            fin = pd.read_parquet(self.fin_path)
            macro = pd.read_parquet(self.macro_path)
            with open(f"{self.tickers_path}", "r") as f:
                tickers = f.read().strip().split("\n")

            X_train, X_val, X_test, y_train, y_val, y_test = _format_tensors_it(fin_data=fin,
                                                                                macro_data=macro,
                                                                                ticker=tickers[self.ticker_idx],
                                                                                lookback=self.lookback,
                                                                                pred_horizon=self.pred_horizon,
                                                                                multistep=self.multistep,
                                                                                pred_target=self.pred_target,
                                                                                multicolinearity_threshold=self.multicolinearity_threshold)

            np.save(f"./cache/it/X_train_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy", X_train)
            np.save(f"./cache/it/X_val_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy", X_val)
            np.save(f"./cache/it/X_test_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy", X_test)
            np.save(f"./cache/it/y_train_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy", y_train)
            np.save(f"./cache/it/y_val_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy", y_val)
            np.save(f"./cache/it/y_test_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy", y_test)


    def setup(self, stage=None):
        self.X_train_tensor = torch.from_numpy(np.load(f"./cache/it/X_train_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy")).float()
        self.X_val_tensor = torch.from_numpy(np.load(f"./cache/it/X_val_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy")).float()
        self.X_test_tensor = torch.from_numpy(np.load(f"./cache/it/X_test_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy")).float()
        self.y_train_tensor = torch.from_numpy(np.load(f"./cache/it/y_train_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy")).float()
        self.y_val_tensor = torch.from_numpy(np.load(f"./cache/it/y_val_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy")).float()
        self.y_test_tensor = torch.from_numpy(np.load(f"./cache/it/y_test_ticker{self.ticker_idx}_{self.data_type}_lookback{self.lookback}_pred_horizon{self.pred_horizon}_multistep{self.multistep}_pred_target{self.pred_target}_multcolinthrsh_{self.multicolinearity_threshold}.npy")).float()

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


def _format_tensors_it(
    fin_data,
    macro_data,
    ticker,
    start_train_at=None,
    lookback=None,
    pred_horizon=1,
    multicolinearity_threshold=None,
    multistep=False,
    debug=False,
    pred_target="price"
):
    if not lookback:
        lookback = pred_horizon * 2
    if multistep and pred_horizon == 1:
        raise ValueError("Multistep only applicable for pred_horizon > 1")

    fin_df = _create_timeseries_features(fin_data.copy().filter(regex=f"^{ticker}_"))
    macro_df = macro_data.copy()

    features = pd.concat([fin_df, macro_df], axis=1)
    if start_train_at:
        features = features.loc[start_train_at:]
    features = features.loc[features[f"{ticker}_CP"] > 0]
    if pred_target == "return":
        target = (features.filter(regex=f"{ticker}_CP").pct_change(pred_horizon) * 100).iloc[pred_horizon:].replace([np.inf, -np.inf], np.nan).fillna(0)
        features = (features.pct_change(pred_horizon) * 100).iloc[pred_horizon:].replace([np.inf, -np.inf], np.nan).fillna(0)
        features = features.loc[:, features.var() != 0]  # Drop features with 0 variance
        features = features.drop(
            columns=[
                f"{ticker}_CP",
                f"{ticker}_OP",
                f"{ticker}_VOL",
                f"{ticker}_OP",
                f"{ticker}_LP",
                f"{ticker}_HP",
            ]
        )
        
        features = pd.DataFrame(minmax_scale(features.values, feature_range=(-1, 1)), columns=features.columns, index=features.index).astype(np.float64)

    else:
        target = features.filter(regex=f"{ticker}_CP")
        features = features.loc[:, features.var() != 0]  # Drop features with 0 variance
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

        # Scale features
        features = pd.DataFrame(robust_scale(features.values), columns=features.columns, index=features.index).astype(np.float64)

    if multicolinearity_threshold:
        corr = features.corr()
        corr = corr.mask(np.tril(np.ones(corr.shape)).astype(bool))
        corr = corr.stack()
        corr = corr[corr > multicolinearity_threshold]
        # Get list of column names to drop
        drop_cols = corr.index.get_level_values(1).drop_duplicates().tolist()
        features = features.drop(columns=drop_cols)
        print(f"Dropped {len(drop_cols)} columns due to multicolinearity")

    X_sequences, y_sequences = [], []

    if multistep:
        for i in tqdm(range(len(features) - lookback)):
            lookback_idx = i + lookback
            pred_idx = lookback_idx + pred_horizon - 1

            if pred_idx > len(features) - 1:
                continue

            X_seq = features.iloc[i:lookback_idx]
            y_seq = target.iloc[lookback_idx : pred_idx + 1]
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

    else:
        for i in tqdm(range(len(features) - lookback)):
            lookback_idx = i + lookback
            pred_idx = lookback_idx + pred_horizon - 1

            if pred_idx > len(features) - 1:
                continue

            X_seq = features.iloc[i:lookback_idx]
            y_seq = target.iloc[pred_idx]
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

    if debug:
        return X_sequences, y_sequences, features, target

    X, y = np.array(X_sequences, dtype=np.float32), np.array(y_sequences, dtype=np.float32)

    # Define split indices (since numpy differs from list slicing)
    test_split = -1
    val_split = -2

    X_train, X_val, X_test = X[:val_split], X[val_split], X[test_split]
    y_train, y_val, y_test = y[:val_split], y[val_split], y[test_split]

    X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])

    return X_train, X_val, X_test, y_train, y_val, y_test
