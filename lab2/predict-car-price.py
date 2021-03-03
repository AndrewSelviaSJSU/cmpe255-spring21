import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt


def linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


# noinspection PyPep8Naming
def prepare_X(df):
    df_num = df[['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

        self.trim()

        np.random.seed(2)

        n = len(self.df)

        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        self.df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()

        y_train_orig = self.df_train.msrp.values
        y_val_orig = df_val.msrp.values
        y_test_orig = df_test.msrp.values

        self.y_train = np.log1p(self.df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        y_test = np.log1p(df_test.msrp.values)

        del self.df_train['msrp']
        del df_val['msrp']
        del df_test['msrp']

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        pass


if __name__ == '__main__':
    car_price = CarPrice()
    X_train = prepare_X(car_price.df_train)
    w_0, w = linear_regression(X_train, car_price.y_train)
    print(w_0)
    print(w)
    y_pred = w_0 + X_train.dot(w)
    print(rmse(car_price.y_train, y_pred))
