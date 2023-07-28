"""
Helper functions for cross-sectorial forecasting models
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tqdm.auto import tqdm


def create_timeseries_features(dataframe):
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


def format_tensors_cs(fin_data, macro_data, tickers, lookback=None, pred_horizon=1, multistep=False):
    if not lookback:
        lookback = pred_horizon * 2
    if multistep and pred_horizon == 1:
        raise ValueError("Multistep only applicable for pred_horizon > 1")
    
    X_train_tensors = []
    X_val_tensors = []
    X_test_tensors = []
    y_train_tensors = []
    y_val_tensors = []
    y_test_tensors = []

    for ticker in tqdm(tickers, desc="Preparing Tensors"):
        fin_df = create_timeseries_features(fin_data.copy().filter(regex=f"^{ticker}_"))
        macro_df = macro_data.copy()

        features = pd.concat([fin_df, macro_df], axis=1)
        features.loc[~(features[f"{ticker}_CP"] > 0), features.columns] = 0 # Make all rows where the target is 0 also 0
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
        
        # Scale features
        features = pd.DataFrame(scale(features.values), columns=features.columns, index=features.index)

        X_sequences, y_sequences = [], []

        if multistep:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features.iloc[i:lookback_idx]
                y_seq = target.iloc[lookback_idx : pred_idx + 1]
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        else:
            for i in range(len(features) - lookback):
                lookback_idx = i + lookback
                pred_idx = lookback_idx + pred_horizon - 1

                if pred_idx > len(features) - 1:
                    continue

                X_seq = features.iloc[i:lookback_idx]
                y_seq = target.iloc[pred_idx]
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

        X_train_tensors.append(X_train)
        X_val_tensors.append(X_val)
        X_test_tensors.append(X_test)
        y_train_tensors.append(y_train)
        y_val_tensors.append(y_val)
        y_test_tensors.append(y_test)

    X_train = np.stack(X_train_tensors)
    X_val = np.stack(X_val_tensors)
    X_test = np.stack(X_test_tensors)
    y_train = np.stack(y_train_tensors)
    y_val = np.stack(y_val_tensors)
    y_test = np.stack(y_test_tensors)

    # Give data shape of (n_samples, channels, timesteps, features)
    X_train = np.transpose(X_train, (1, 0, 2, 3))
    X_val = np.transpose(X_val, (1, 0, 2, 3))
    X_test = np.transpose(X_test, (1, 0, 2, 3))

    y_train = np.transpose(y_train.squeeze(), (1, 0))
    y_val = np.transpose(y_val, (1, 0))
    y_test = np.transpose(y_test, (1, 0))

    return X_train, X_val, X_test, y_train, y_val, y_test
