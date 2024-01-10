import pandas as pd

def data_shift(train, test, lags=7):
    train_shifted =  pd.concat([train.shift(lags-i) for i in range(lags+1)], axis=1)
    train_shifted.columns = [f't-{lags-i}' for i in range(lags)] + ['t']
    train_shifted = train_shifted.iloc[lags:]

    test_shifted =  pd.concat([test.shift(lags-i) for i in range(lags+1)], axis=1)
    test_shifted.columns = [f't-{lags-i}' for i in range(lags)] + ['t']
    test_shifted = test_shifted.iloc[lags:]
    
    return train_shifted, test_shifted