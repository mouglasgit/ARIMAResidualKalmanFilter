
import numpy as np
import pandas as pd

def load_series_airlines():
    train_size = 110
    test_size = 143
    
    path = './data'
    data = pd.read_csv(f"{path}/airlines2.txt", header=None).values.astype(np.float64).ravel()
    train, test = data[:train_size], data[train_size:train_size+test_size]
    
    return train, test

def load_dataset_conv_cr():
    train_size = 595
    test_size = 743
    
    path = './data'
    data = pd.read_csv(f"{path}/coloradoRiver.txt", header=None).values.astype(np.float64).ravel()
    train, test = data[:train_size], data[train_size:train_size+test_size]
    
    return train, test

def load_dataset_conv_sun():
    train_size = 230
    test_size = 288
    
    path = './data'
    df   = pd.read_csv(f"{path}/Sunspot.txt", header=None)
    data = df.values.astype(np.float32).ravel()
    
    train, test = data[:train_size], data[train_size:train_size+test_size]
    return train, test


def load_dataset_conv_lynx():
    train_size = 100
    test_size = 113
    
    path = './data'
    df   = pd.read_csv(f"{path}/lynx.txt", header=None)
    data = df.values.astype(np.float32).ravel()
    data = np.log10(data)
    
    train, test = data[:train_size], data[train_size:train_size+test_size]
    return train, test
    