import numpy as np
import pandas as pd
from utils import main

class Kolmogorov_Smirnov:
    def __init__(self, x, y):
        self.x = np.array(x)      # shape: (n1, d)
        self.y = np.array(y)      # shape: (n2, d)
        self.col_list = x.columns

    def ks_test(self):
        # Sort each column
        x_sorted = np.sort(self.x, axis=0)  # (n1, d)
        y_sorted = np.sort(self.y, axis=0)  # (n2, d)

        # Combine unique values column-wise (broadcastable trick)
        comb = np.sort(
            np.unique(
                np.concatenate([x_sorted, y_sorted], axis=0),
                axis=0
            ),
            axis=0
        )  # shape: (k, d)

        # Reshape for broadcasting
        # x_sorted: (n1, d) -> (n1, 1, d)
        # y_sorted: (n2, 1, d)
        # comb:     (1, k, d)
        x_b = x_sorted[:, None, :]
        y_b = y_sorted[:, None, :]
        comb_b = comb[None, :, :]

        # ECDF using broadcasting: count(x <= comb)
        Fx = (x_b <= comb_b).sum(axis=0) / x_sorted.shape[0]  # (k, d)
        Fy = (y_b <= comb_b).sum(axis=0) / y_sorted.shape[0]  # (k, d)

        # KS statistic per column
        D = np.max(np.abs(Fx - Fy), axis=0)  # (d,)

        # return in a column-name dict
        return dict(zip(self.col_list, D))
if __name__ == '__main__':
    source=pd.read_csv("data\source.csv")
    stream=pd.read_csv("data\stream.csv")
    target_column="Outlet_Type"
    source,_,stream,_ = main(source, stream, target_column)
    ks = Kolmogorov_Smirnov(source, stream)
    print(ks.ks_test())
    # print(ks.x)
    # x, y = ks.ks_test()
    # print(x,y)