import os
import pickle

import numpy as np
from epftoolbox.data import read_data, scaling

from data.loader import MARKETS

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"
VAL_HOURS = 42 * 7 * 24


def process(market):
    df_train, df_test = read_data(path=RAW_DIR, dataset=market)
    df_val = df_train.iloc[-VAL_HOURS:]
    df_train = df_train.iloc[:-VAL_HOURS]

    (train, val, test), scaler = scaling(
        [df_train.values, df_val.values, df_test.values], normalize="Median")

    return {
        "train": np.ascontiguousarray(train, dtype=np.float32),
        "val": np.ascontiguousarray(val, dtype=np.float32),
        "test": np.ascontiguousarray(test, dtype=np.float32),
        "dates_train": df_train.index.values,
        "dates_val": df_val.index.values,
        "dates_test": df_test.index.values,
        "scaler": scaler,
    }


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for market in MARKETS:
        d = process(market)
        with open(os.path.join(PROCESSED_DIR, f"{market}_series.pkl"), "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{market}: train={len(d['train'])}h val={len(d['val'])}h "
              f"test={len(d['test'])}h")


if __name__ == "__main__":
    main()
