"""
Helper functions for iterative forecasting models
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


def format_tensors_it(
    fin_data,
    macro_data,
    ticker,
    start_train_at=None,
    lookback=None,
    pred_horizon=1,
    multicolinearity_threshold=None,
    multistep=False,
    debug=False,
):
    if not lookback:
        lookback = pred_horizon * 2
    if multistep and pred_horizon == 1:
        raise ValueError("Multistep only applicable for pred_horizon > 1")

    fin_df = create_timeseries_features(fin_data.copy().filter(regex=f"^{ticker}_"))
    macro_df = macro_data.copy()

    features = pd.concat([fin_df, macro_df], axis=1)
    if start_train_at:
        features = features.loc[start_train_at:]
    features = features.loc[features[f"{ticker}_CP"] > 0]
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
    features = pd.DataFrame(scale(features.values), columns=features.columns, index=features.index)

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
