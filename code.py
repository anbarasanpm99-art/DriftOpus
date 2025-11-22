import pandas as pd
import numpy as np
from utils import main

class Kolmogorov_Smirnov():
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.col_list=x.columns

    def ks_test(self):

        # # Find the length of the arrays
        # n1=len(self.x)
        # n2=len(self.y)

        #combine source and stream to a single array
        comb=np.unique(np.concatenate([self.x, self.y]), axis = 1)

        self.x = self.x.T
        self.y = self.y.T

        comb = comb.T

        comb=comb[:,:,None]
        self.x=self.x[:,None,:]
        self.y=self.y[:,None,:]


     
        
        # Ratio using array broadcasting
        F_n1=(self.x <= comb).sum(axis=2)/self.x.shape[0]
        F_n2=(self.y <= comb).sum(axis=2)/self.y.shape[0]

        print(F_n1)
        print(F_n2)
        # F_n1 = self.distribution_function(self.x, comb)
        # F_n2 = self.distribution_function(self.y, comb)
        # Maximum Difference between F_n1 and F_n2
        D=np.max(np.abs(F_n1-F_n2), axis = 0)
        
        return D


    def distribution_function(self, arr, comb):
        arr = arr.T
        comb = comb.T
        arr = arr[:,None,:]
        comb = comb[:,:,None]
        distribution = (arr >= comb).sum(axis=2)/len(arr)
        return distribution

        # counts = (arr[:, :, None] >= arr[:, None, :]).sum(axis=2)
        # distribution = counts/arr.shape[1]
        # return distribution
        
    


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