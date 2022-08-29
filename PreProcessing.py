import os
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler




def load_NCMAPSS_data():
    '''
        Function to load N-CMAPPS dataset from specified file path

        returns:
        (dev_data,y_dev) : to be used for training 
        dev_data : Data_frame object that contains the sensosrs readings and other auxiliary variables
        y_dev : numpy array contains the  corresponding to targets ,

        
        returns:
        (test_data,y_test) : to be used for testing 
        dev_data : Data_frame object that contains the sensosrs readings and other auxiliary variables
        y_dev : numpy array contains the  corresponding to targets ,
    '''
    #change the path to specified path where the data uploaded 

    file_path = '/content/drive/MyDrive/Colab Notebooks/N-CMAPSS_DS08a-009.h5'

    # Load data
    with h5py.File(file_path, 'r') as hdf:
        # Development set
        W_dev = np.array(hdf.get('W_dev'))             # W
        X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v
        T_dev = np.array(hdf.get('T_dev'))             # T
        Y_dev = np.array(hdf.get('Y_dev'))             # RUL  
        A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))           # W
        X_s_test = np.array(hdf.get('X_s_test'))       # X_s
        X_v_test = np.array(hdf.get('X_v_test'))       # X_v
        T_test = np.array(hdf.get('T_test'))           # T
        Y_test = np.array(hdf.get('Y_test'))           # RUL  
        A_test = np.array(hdf.get('A_test'))           # Auxiliary
        
        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))  
        X_v_var = np.array(hdf.get('X_v_var')) 
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))  
        X_v_var = list(np.array(X_v_var, dtype='U20')) 
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))


    #Put the data into a DataFrame format
    dev_data = DataFrame(data=np.concatenate((A_dev,X_s_dev,X_v_dev),axis=1),
                           columns=(A_var+X_s_var+X_v_var))

    test_data = DataFrame(data=np.concatenate((A_test,X_s_test,X_v_test),axis=1),
                          columns=(A_var+X_s_var+X_v_var))

    return (dev_data,Y_dev),(test_data,Y_test)


#plotting sensors reading we took the mean of each cycle

def plot_senors_reading(data):
    '''
    Plot the sensors readings
    '''
    data_mean = DataFrame(columns=data.columns)
    n_units = data.unit.unique()

    for unit in n_units:
        unit_data = data[data.unit==unit]#Extract the the data of one unit
        t_eof = int(unit_data.cycle.max())
        cycle_mean = np.zeros((t_eof,unit_data.shape[1]))
        for c in range(t_eof):
            mean = np.array(unit_data[unit_data.cycle==c].mean(axis=0))
            mean = mean.reshape(1,unit_data.shape[1])
            cycle_mean[c] = mean # the mean of the measurments at the c-th cycle
        cycle_mean = DataFrame(data = cycle_mean,columns=data.columns)
        data_mean = pd.concat([data_mean,cycle_mean])
    data_mean.dropna(subset=data_mean.columns)
    data_mean.drop(0)


    
    #Selected Sensors to Plot
    col_list = data.columns[4:18]
            
    fig, ax = plt.subplots(ncols=7, nrows =2, figsize=(30, 10))
    ax = ax.ravel()
    for i, item in enumerate(col_list):
        data_mean.groupby('unit').plot(kind='line', x = "cycle", y = item, ax=ax[i])
        ax[i].get_legend().remove()
        ax[i].title.set_text(item)
    plt.subplots_adjust(top = 0.99, bottom = 0.01, hspace = 0.3, wspace = 0.2)
    plt.show()


def plot_virtual_senors_reading(data):
    # Extract the mean of the cylcles of each unit
    data_mean = DataFrame(columns=data.columns)
    n_units = data.unit.unique()

    for unit in n_units:
        unit_data = data[data.unit==unit]#Extract the the data of one unit
        t_eof = int(unit_data.cycle.max())
        cycle_mean = np.zeros((t_eof,unit_data.shape[1]))
        for c in range(t_eof):
            mean = np.array(unit_data[unit_data.cycle==c].mean(axis=0))
            mean = mean.reshape(1,unit_data.shape[1])
            cycle_mean[c] = mean # the mean of the measurments at the c-th cycle
        cycle_mean = DataFrame(data = cycle_mean,columns=data.columns)
        data_mean = pd.concat([data_mean,cycle_mean])
    data_mean.dropna(subset=data_mean.columns)
    data_mean.drop(0)


    
    #plotting the vertual sensors readings
    col_list = data.columns[18:32]
            
    fig, ax = plt.subplots(ncols=7, nrows =2, figsize=(30, 10))
    ax = ax.ravel()
    for i, item in enumerate(col_list):
        data_mean.groupby('unit').plot(kind='line', x = "cycle", y = item, ax=ax[i])
        ax[i].get_legend().remove()
        ax[i].title.set_text(item)
    plt.subplots_adjust(top = 0.99, bottom = 0.01, hspace = 0.3, wspace = 0.2)
    plt.show()


def data_normalization(X_train,X_test):
    '''
    Parameters : 
    X_train: training features to be normalized 
    X_test : testing features to be normalized using the mean and std of the training set
    '''
    import warnings
    warnings.filterwarnings("ignore")
    Scaler = StandardScaler()
    X_train.iloc[:,2:] =Scaler.fit_transform(X_train.iloc[:,2:])
    X_test.iloc[:,2:] =Scaler.transform(X_test.iloc[:,2:])

    return X_train,X_test