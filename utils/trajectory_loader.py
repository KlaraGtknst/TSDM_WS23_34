import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch

class SingleTimeSeriesData(Dataset):
    def __init__(self, X, y, n):
        self.X = X.values
        self.y = y.values
        self.index = y.index
        self.n_steps = n
        
    def __len__(self):
        return len(self.y) - self.n_steps
    
    def __getitem__(self, idx):
        return self.X[idx:idx + self.n_steps], self.y[idx:idx + self.n_steps]     
    

class TimeSeriesData(Dataset):
    def __init__(self, X, y, sw):
        """
        TimeSeries Pytorch Dataset
        
        Parameters
        ----------
        X : Pandas Dataframe
            Input Features
            
        y : Pandas Dataframe
            Labels
            
        sw : str
            Windowsize in s, ms, us. examples: <'2s'> <20ms'> <'10us'>
        
        """
        self.X = X
        self.y = y
        self.feat_window_length = sw
        self.n_steps = calc_window_size(X.index.values[0][3], X.index.values[1][3], self.feat_window_length)
        
        # Create for each Ts Group a PyTorch Dataset Object
        index_names = self.X.index.names[:-1]
        self.ts_list = []
        for name, group in self.X.groupby(level=index_names):
            ts_data = SingleTimeSeriesData(group, self.y.loc[name], self.n_steps)
            self.ts_list.append(ts_data)
            
        # Concatenate all objects
        self.ts = ConcatDataset(self.ts_list)
        self.len_of_series = len(self.ts)
        self.index_values = np.arange(self.len_of_series)
        
    def get_ts_list(self):
        return self.ts_list
        
    def __len__(self):
        return self.len_of_series
    
    def __getitem__(self, idx):
        x_seq, y_seq = self.ts[self.index_values[idx]]
        return torch.Tensor(x_seq).float(), torch.Tensor(y_seq).float()


def calc_window_size(timestamp_1, timestamp_2, window_length):
    """ Calculate the window length in units i.e. data points.
        Therefore the inherent sampling frequency of the data is estimated given to distinct timestamps
        of two subsequent data points.

        Parameters:
        -----------
        timestamp_1: Timestamp, DateTimeIndex
           First value which to use for frequency calculation.

        timestamp_2: Timestamp, DateTimeIndex
           Second value which to use for frequency calculation.

        window_length: string
           The length of the sliding window.

        Returns:
        --------
        windows_size: int
            Number of points in window, rounded down to integer.
    """
    delta = round(pd.Timedelta(timestamp_2 - timestamp_1).microseconds / 1000)

    #if pd.Timedelta(window_length).microseconds // 1000 % delta != 0:
    #    raise RuntimeError(f'window_length ({window_length}) not multiple of timestamp delta ({delta}ms).')

    # calculate frequency of input data (approx. from distance of first to timestamps)
    loc_freq = 1.0 / (float(delta) / 1000.0)
    # calculate window length in seconds represented as float
    loc_window_length = pd.Timedelta(window_length).total_seconds()

    # calculate window length ins units i.e. data points.
    window_size = int(loc_freq * loc_window_length)
    #window_size = int((loc_freq * loc_window_length) + 1)

    return window_size
