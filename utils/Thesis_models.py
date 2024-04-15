import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold

from lmmnn.nn import reg_nn_ohe_or_ignore, reg_nn_lmm, reg_nn_embed, reg_nn_rnn
from lmmnn.simulation import Count
from lmmnn.utils import *
import gc
import tensorflow.keras.backend as K

import tensorflow as tf

import logging
logging.disable(logging.INFO)
from merf import MERF

from armed.models.mlp_classifiers_reg import MixedEffectsMLP

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def Linear_Exclude_Group(df):
    
    dataframe = df.copy()
    
    column_to_select = [i for i in dataframe.columns if i.startswith('f')] + ["y"]
    dataframe = dataframe[column_to_select]
    
    start = time.time()
    X,y = dataframe.drop('y', axis=1), dataframe.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    end = time.time()
    mse_Linear_Exclude_Group = np.mean((y_pred - y_test)**2)
    
    return mse_Linear_Exclude_Group, end-start, y_test, y_pred

def Linear_Include_Group(df,to_drop = None):
    
    dataframe = df.copy()
    
    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)
    
    start = time.time()
    X,y = dataframe.drop('y', axis=1), dataframe.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    end = time.time()
    mse_Linear_Include_Group = np.mean((y_pred - y_test)**2)
    
    return mse_Linear_Include_Group, end-start, y_test, y_pred

def LinearOHE(df, to_drop = None):
    
    dataframe = df.copy()
    
    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)

    column_to_encode = [col for col in dataframe.columns if col if col.startswith('g')]
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(dataframe[column_to_encode])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_to_encode))
    dataframe = pd.concat([dataframe, encoded_df], axis=1)
    dataframe.drop(column_to_encode, axis=1, inplace=True)

    start = time.time()
    X,y = dataframe.drop('y', axis=1), dataframe.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end = time.time()
    mse_linearohe = np.mean((y_pred - y_test)**2)
    return mse_linearohe, end-start, y_test, y_pred 

# def MixedLM(df, mode, to_drop = None ):
    
#     dataframe = df.copy()    
#     allowed_modes = ['intercept', 'slope', 'both']
    
#     if mode not in allowed_modes:
#         raise ValueError(f"Invalid mode '{mode}'. Please choose one of {', '.join(allowed_modes)}.")
#     if to_drop:
#         dataframe = dataframe.drop(to_drop, axis=1)
        
#     X, y = dataframe.drop('y', axis=1), dataframe['y']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
#     x_cols = [col for col in X.columns if col if not col.startswith('g')]    
#     g_cols = [col for col in X.columns if col if col.startswith('g')]
    
#     scaler = StandardScaler()
#     X_train[x_cols] = scaler.fit_transform(X_train[x_cols])
#     X_test[x_cols]  = scaler.transform(X_test[x_cols])
#     data_train = pd.concat([X_train, y_train], axis=1)
    
#     start = time.time()
    
#     if mode == 'intercept':
#         md = sm.MixedLM.from_formula(f"y ~ {' + '.join([i for i in X.columns if i.startswith('f')])} + (1|{g_cols[0]})", \
#          data_train, groups = str(g_cols[0]))
        
#     if mode == 'slope':
#         md = sm.MixedLM.from_formula(f"y ~ {' + '.join([i for i in X.columns if i.startswith('f')])}", \
#          data_train, re_formula = '~'+'+'.join([i for i in df.columns if i.startswith('f')]), groups = str(g_cols[0]))
        
#     if mode == 'both':
#         md = sm.MixedLM.from_formula(f"y ~ {' + '.join([i for i in X.columns if i.startswith('f')])} + (1|{g_cols[0]})", \
#          data_train, re_formula = '~'+'+'.join([i for i in df.columns if i.startswith('f')]), groups = str(g_cols[0]))
    
#     mdf = md.fit()
    
#     preds = mdf.predict(X_test)
    
#     for i in X_test.index:

#         preds[i] = preds[i] + mdf.random_effects[X_test.at[i,str(g_cols[0])]][0]

#         try:
#             for itr, f in enumerate(x_cols):
#                 preds[i] = preds[i] + mdf.random_effects[X_test.at[i,str(g_cols[0])]][itr+1] * X_test.at[i,str(f)]
#         except (KeyError, IndexError): pass
    
#     end = time.time()
    
#     MSE = np.mean((preds - y_test)**2)
    
#     return MSE, end-start, y_test, preds

def MixedLM(df, mode, to_drop=None, k=3):
    
    dataframe = df.copy()
    allowed_modes = ['intercept', 'slope', 'both']
    
    if mode not in allowed_modes:
        raise ValueError(f"Invalid mode '{mode}'. Please choose one of {', '.join(allowed_modes)}.")

    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)
        
    stratify_col = [col for col in dataframe.columns if col if col.startswith('g')][0]

    X, y = dataframe.drop('y', axis=1), dataframe['y']
    stratify_groups = dataframe[stratify_col]
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    x_cols = [col for col in X.columns if not col.startswith('g')]    
    g_cols = [col for col in X.columns if col.startswith('g')]
    scaler = StandardScaler()
    
    total_mse = 0
    total_time = 0
    y_test_list = []
    preds_list = []

    for train_index, test_index in skfold.split(X, stratify_groups):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train[x_cols] = scaler.fit_transform(X_train[x_cols])
        X_test[x_cols] = scaler.transform(X_test[x_cols])
        data_train = pd.concat([X_train, y_train], axis=1)

        start = time.time()
        if mode == 'intercept':
            md = sm.MixedLM.from_formula(f"y ~ {' + '.join(x_cols)} + (1|{g_cols[0]})", data_train, \
                                         groups=data_train[g_cols[0]])
        elif mode == 'slope':
            md = sm.MixedLM.from_formula(f"y ~ {' + '.join(x_cols)}", data_train, \
                                         re_formula='~' + '+'.join(x_cols), groups=data_train[g_cols[0]])
        elif mode == 'both':
            md = sm.MixedLM.from_formula(f"y ~ {' + '.join(x_cols)} + (1|{g_cols[0]})", data_train, \
                                         re_formula='~' + '+'.join(x_cols), groups=data_train[g_cols[0]])

        mdf = md.fit()
        preds = mdf.predict(X_test)
        
        for i in X_test.index:
            try:
                if mode in ['intercept', 'both']:
                    preds[i] = preds[i] + mdf.random_effects[X_test.at[i, g_cols[0]]][0]
                for itr, f in enumerate(x_cols):
                    preds[i] = preds[i] + mdf.random_effects[X_test.at[i, g_cols[0]]][itr + 1] * X_test.at[i, f]
            except (KeyError, IndexError):
                pass

        end = time.time()
        total_time += end - start
        MSE = np.mean((preds - y_test) ** 2)
        total_mse += MSE
        y_test_list.extend(y_test)
        preds_list.extend(preds)

    avg_mse = total_mse/k
    avg_time = total_time/k

    return avg_mse, avg_time, y_test_list, preds_list

def LMMNN(df, use_OHE = False, to_drop = None):
    
    dataframe = df.copy()
    
    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)
    
    if use_OHE == True:
        
        column_to_encode = [col for col in dataframe.columns if col if col.startswith('g')]
        encoder = OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(dataframe[column_to_encode])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_to_encode))
        final_df = pd.concat([dataframe, encoded_df], axis=1)
        dataframe = final_df.copy()
        
        column_to_encode = [col for col in dataframe.columns if col if col.startswith('g')]
        dataframe.rename(columns={f'{col}': f'z{i}' for i, col in enumerate(column_to_encode)}, inplace=True)
        
    else:
        dataframe.rename(columns={[col for col in dataframe.columns if col if col.startswith('g')][0]: 'z0'}, inplace=True)

    z_cols = sorted([col for col in dataframe.columns if col.startswith('z')], key=lambda x: int(x[1:]))

    mode = 'intercepts'
    n_cats = [len(dataframe[i].unique()) for i in z_cols]
    qs = n_cats
    batch_size = 100
    epochs = 500
    patience = 10
    n_sig2bs = len(n_cats)
    n_sig2bs_spatial = 0
    est_cors = []
    n_neurons = [50,25,12,6]
    activation = 'linear'
    dropout = []
    spatial_embedded_neurons = []
    dist_matrix = None
    q_spatial = None

    print('n_uniques: ', n_cats)

    X, y = dataframe.drop('y', axis=1), dataframe['y']
    x_cols = [col for col in X.columns if col if not col.startswith('z')]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train[x_cols] = scaler.fit_transform(X_train[x_cols])
    X_test[x_cols]  = scaler.transform(X_test[x_cols])

    def reg_nn(X_train, X_test, y_train, y_test, n_cats, batch=batch_size, epochs=epochs, \
               patience=patience, reg_type='lmm', verbose=False):

        start = time.time()
        if reg_type == 'lmm':
            model_fitted, history, y_pred, sigmas, _, _, n_epochs = reg_nn_lmm(X_train, X_test, y_train, y_test, n_cats, \
                                                                               q_spatial, x_cols, batch, epochs, patience,
                                                     n_neurons, dropout, activation,
                                                     mode=mode, n_sig2bs=n_sig2bs, n_sig2bs_spatial=n_sig2bs_spatial,
                                                     est_cors=est_cors, dist_matrix=dist_matrix,
                                                     spatial_embed_neurons=spatial_embedded_neurons, verbose=verbose, log_params=False)
        else:
            raise ValueError(reg_type + 'is an unknown reg_type')
        end = time.time()
        gc.collect()
        K.clear_session()
        mse = np.mean((y_pred - y_test)**2)
        return mse, sigmas, n_epochs, end - start, y_pred, model_fitted

    mse_lmm, sigmas, n_epochs_lmm, time_lmm,y_pred, model= reg_nn(X_train, X_test, y_train, y_test, n_cats, reg_type='lmm', verbose=False)

    return mse_lmm, time_lmm, y_test, y_pred

def MERForest(df, max_iterations = 20,threshold = 0.1, to_drop = None):
    
    dataframe = df.copy()
    
    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)
        
    start = time.time()
    RS = 42
    test_ratio = 0.33
    val_ratio = 0.1

    X,y = dataframe.drop('y', axis=1), dataframe.y

    test_indices = X.sample(frac=test_ratio, random_state=RS).index
    split = [(np.array(list(set(X.index).difference(test_indices))), np.array(test_indices))]

    for num, (train_indices, test_indices) in enumerate(split):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RS)
    
    x_cols = [col for col in X.columns if col if not col.startswith('g')]
    g_cols = [col for col in X.columns if col if col.startswith('g')]
    scaler = StandardScaler()
    X_train[x_cols] = scaler.fit_transform(X_train[x_cols])
    X_val[x_cols]   = scaler.fit_transform(X_val[x_cols])
    X_test[x_cols]  = scaler.transform(X_test[x_cols])
    
    np.random.seed(100)
    clusters_train = X_train[g_cols[0]]
    Z_train = np.ones(shape=(X_train.shape[0],1))
    clusters_val = X_val[g_cols[0]]
    Z_val = np.ones(shape=(X_val.shape[0],1))
    
    start = time.time()
    merf = MERF(max_iterations = max_iterations, gll_early_stop_threshold=threshold)
    merf.fit(X_train, Z_train, clusters_train, y_train, X_val, Z_val, clusters_val, y_val)

    clusters_test = X_test[g_cols[0]]
    X_test_merf = X_test.drop(g_cols,axis=1)
    Z_test = np.ones(shape=(X_test.shape[0],1))
    y_pred = merf.predict(X_test, Z_test, clusters_test)
    end = time.time()
    
    mse_merf = np.mean((y_pred - y_test)**2)
    
    return mse_merf, end-start, y_test, y_pred

def ARMED(df, to_drop = None, epochs = 500):
    
    dataframe = df.copy()
    
    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)
    
    RS = 42
    test_ratio = 0.33
    val_ratio = 0.1
    np.random.seed(100)

    X,y = dataframe.drop('y', axis=1), dataframe.y

    test_indices = X.sample(frac=test_ratio, random_state=RS).index
    split = [(np.array(list(set(X.index).difference(test_indices))), np.array(test_indices))]

    for num, (train_indices, test_indices) in enumerate(split):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RS)
        
    g_cols = [col for col in X.columns if col if col.startswith('g')]

    Z_train = X_train[g_cols[0]]
    Z_val = X_val[g_cols[0]]
    Z_test = X_test[g_cols[0]]

    X_train.drop(g_cols[0],axis=1,inplace=True)
    X_val.drop(g_cols[0],axis=1,inplace=True)
    X_test.drop(g_cols[0],axis=1,inplace=True)

    Z_train_ohe = tf.one_hot(Z_train.values.ravel(),Z_train.nunique()).numpy()
    Z_val_ohe = tf.one_hot(Z_val.values.ravel(),Z_train.nunique()).numpy()
    Z_test_ohe = tf.one_hot(Z_test.values.ravel(),Z_train.nunique()).numpy()

    dictBuild = {'n_features': X_train.shape[1],
                 'n_clusters': Z_train.nunique(),
                 'adversary_layer_units': [4,8,4],# [4,4]
                 'slope_posterior_init_scale': 0.3, # 0.3
                 'intercept_posterior_init_scale': 0.1, #0.1
                 'slope_prior_scale': 0.3, #0.3
                 'intercept_prior_scale': 0.1, #0.1
                 'kl_weight': 0.00001} # 0.00001

    dictCompile = {'loss_class_fe_weight': 1, #1.0 
                   'loss_gen_weight': 0.5,      #0.5
                   'loss_class_me_weight': 1, #1.0
                   'metric_class_me': tf.keras.metrics.MeanSquaredError(name='MAE-class_me'),
                   'metric_class_fe': tf.keras.metrics.MeanSquaredError(name='MAE-class_fe'),
                   'metric_adv': tf.keras.metrics.MeanSquaredError(name='MAE-adv'),
                   'loss_class': tf.keras.losses.MeanSquaredError(name='MSE-loss_class'),
                   'loss_adv': tf.keras.losses.MeanSquaredError(name='MSE-loss_adv')}

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=3e-4, decay=1e-6)
    lsCallbacks = []

    start = time.time()
    
    model_armed = MixedEffectsMLP(**dictBuild)
    model_armed.compile(**dictCompile)

    log = model_armed.fit((X_train,Z_train_ohe), y_train,
                    validation_data=((X_val,Z_val_ohe), y_val),
                    callbacks=lsCallbacks,
                    epochs=epochs,
#                     batch_size=batch_size,                
                    verbose=0)

    pred = model_armed.predict((X_test,Z_test_ohe))
    end = time.time()
    
    mse_armed = mse(y_test, pred[0])
    
    return mse_armed, end-start, y_test, pred[0]



############ Train-Test Performance ##############



def MixedLM_train_test(df, mode, calculate_for = None):
    """
    output_array = mse, time, y_test, y_pred
    
    """
    
    dataframe = df.copy()    
    allowed_modes = ['intercept', 'slope', 'both']
    
    if mode not in allowed_modes:
        raise ValueError(f"Invalid mode '{mode}'. Please choose one of {', '.join(allowed_modes)}.")
    if calculate_for:
        dataframe = dataframe.drop([col for col in dataframe.columns if col if col.startswith('g') \
                                                                and col != calculate_for], axis=1)
        
    X, y = dataframe.drop('y', axis=1), dataframe['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    x_cols = [col for col in X.columns if col if not col.startswith('g')]    
    g_cols = [col for col in X.columns if col if col.startswith('g')]
    
    scaler = StandardScaler()
    X_train[x_cols] = scaler.fit_transform(X_train[x_cols])
    X_test[x_cols]  = scaler.transform(X_test[x_cols])
    data_train = pd.concat([X_train, y_train], axis=1)
    
    start = time.time()
    
    if mode == 'intercept':
        md = sm.MixedLM.from_formula(f"y ~ {' + '.join([i for i in X.columns if i.startswith('f')])} + (1|{g_cols[0]})", \
         data_train, groups = str(g_cols[0]))
        
    if mode == 'slope':
        md = sm.MixedLM.from_formula(f"y ~ {' + '.join([i for i in X.columns if i.startswith('f')])}", \
         data_train, re_formula = '~'+'+'.join([i for i in df.columns if i.startswith('f')]), groups = str(g_cols[0]))
        
    if mode == 'both':
        md = sm.MixedLM.from_formula(f"y ~ {' + '.join([i for i in X.columns if i.startswith('f')])} + (1|{g_cols[0]})", \
         data_train, re_formula = '~'+'+'.join([i for i in df.columns if i.startswith('f')]), groups = str(g_cols[0]))
    
    mdf = md.fit()
    
    preds = mdf.predict(X_test)
    
    for i in X_test.index:

        preds[i] = preds[i] + mdf.random_effects[X_test.at[i,str(g_cols[0])]][0]

        try:
            for itr, f in enumerate(x_cols):
                preds[i] = preds[i] + mdf.random_effects[X_test.at[i,str(g_cols[0])]][itr+1] * X_test.at[i,str(f)]
        except (KeyError, IndexError): pass
    
    end = time.time()
    
    mse_train = np.mean((mdf.fittedvalues - data_train.y)**2) 
    mse_test = np.mean((preds - y_test)**2)
    
    return mse_train, mse_test, end-start, y_test, preds

def ARMED_train_test(df, to_drop = None, epochs = 500):
    
    dataframe = df.copy()
    
    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)
    
    RS = 42
    test_ratio = 0.33
    val_ratio = 0.1
    np.random.seed(100)

    X,y = dataframe.drop('y', axis=1), dataframe.y

    test_indices = X.sample(frac=test_ratio, random_state=RS).index
    split = [(np.array(list(set(X.index).difference(test_indices))), np.array(test_indices))]

    for num, (train_indices, test_indices) in enumerate(split):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RS)
        
    g_cols = [col for col in X.columns if col if col.startswith('g')]

    Z_train = X_train[g_cols[0]]
    Z_val = X_val[g_cols[0]]
    Z_test = X_test[g_cols[0]]

    X_train.drop(g_cols[0],axis=1,inplace=True)
    X_val.drop(g_cols[0],axis=1,inplace=True)
    X_test.drop(g_cols[0],axis=1,inplace=True)

    Z_train_ohe = tf.one_hot(Z_train.values.ravel(),Z_train.nunique()).numpy()
    Z_val_ohe = tf.one_hot(Z_val.values.ravel(),Z_train.nunique()).numpy()
    Z_test_ohe = tf.one_hot(Z_test.values.ravel(),Z_train.nunique()).numpy()

    dictBuild = {'n_features': X_train.shape[1],
                 'n_clusters': Z_train.nunique(),
                 'adversary_layer_units': [4,8,4],# [4,4]
                 'slope_posterior_init_scale': 0.3, # 0.3
                 'intercept_posterior_init_scale': 0.1, #0.1
                 'slope_prior_scale': 0.3, #0.3
                 'intercept_prior_scale': 0.1, #0.1
                 'kl_weight': 0.00001} # 0.00001

    dictCompile = {'loss_class_fe_weight': 1, #1.0 
                   'loss_gen_weight': 0.5,      #0.5
                   'loss_class_me_weight': 1, #1.0
                   'metric_class_me': tf.keras.metrics.MeanSquaredError(name='MAE-class_me'),
                   'metric_class_fe': tf.keras.metrics.MeanSquaredError(name='MAE-class_fe'),
                   'metric_adv': tf.keras.metrics.MeanSquaredError(name='MAE-adv'),
                   'loss_class': tf.keras.losses.MeanSquaredError(name='MSE-loss_class'),
                   'loss_adv': tf.keras.losses.MeanSquaredError(name='MSE-loss_adv')}

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=3e-4, decay=1e-6)
    lsCallbacks = []

    start = time.time()
    
    model_armed = MixedEffectsMLP(**dictBuild)
    model_armed.compile(**dictCompile)

    log = model_armed.fit((X_train,Z_train_ohe), y_train,
                    validation_data=((X_val,Z_val_ohe), y_val),
                    callbacks=lsCallbacks,
                    epochs=epochs,
#                     batch_size=batch_size,                
                    verbose=0)

    pred = model_armed.predict((X_test,Z_test_ohe))
    pred_train = model_armed.predict((X_train,Z_train_ohe))
    
    end = time.time()
    
    mse_armed = mse(y_test, pred[0])
    mse_train = mse(y_train, pred_train[0])
    
    return mse_train, mse_armed, end-start, y_test, pred[0]

def MERForest_train_test(df, max_iterations = 20,threshold = 0.1, to_drop = None):
    
    dataframe = df.copy()
    
    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)
        
    start = time.time()
    RS = 42
    test_ratio = 0.33
    val_ratio = 0.1

    X,y = dataframe.drop('y', axis=1), dataframe.y

    test_indices = X.sample(frac=test_ratio, random_state=RS).index
    split = [(np.array(list(set(X.index).difference(test_indices))), np.array(test_indices))]

    for num, (train_indices, test_indices) in enumerate(split):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RS)
    
    x_cols = [col for col in X.columns if col if not col.startswith('g')]
    g_cols = [col for col in X.columns if col if col.startswith('g')]
    scaler = StandardScaler()
    X_train[x_cols] = scaler.fit_transform(X_train[x_cols])
    X_val[x_cols]   = scaler.fit_transform(X_val[x_cols])
    X_test[x_cols]  = scaler.transform(X_test[x_cols])
    
    X_train_new = X_train.copy()
    y_train_new = y_train.copy()
    
    np.random.seed(100)
    clusters_train = X_train[g_cols[0]]
    Z_train = np.ones(shape=(X_train.shape[0],1))
    clusters_val = X_val[g_cols[0]]
    Z_val = np.ones(shape=(X_val.shape[0],1))
    
    start = time.time()
    merf = MERF(max_iterations = max_iterations, gll_early_stop_threshold=threshold)
    merf.fit(X_train, Z_train, clusters_train, y_train, X_val, Z_val, clusters_val, y_val)
    
    # Training pred
    clusters_train_new = X_train_new[g_cols[0]]
    Z_train_new = np.ones(shape=(X_train_new.shape[0],1))
    y_pred_train = merf.predict(X_train_new, Z_train_new, clusters_train_new)

    # Testing pred
    clusters_test = X_test[g_cols[0]]
    Z_test = np.ones(shape=(X_test.shape[0],1))
    y_pred = merf.predict(X_test, Z_test, clusters_test)
    end = time.time()
    
    mse_train = np.mean((y_pred_train - y_train_new)**2)
    mse_test = np.mean((y_test- y_pred)**2)
    
    return mse_train, mse_test, end-start, y_test, y_pred 

def LMMNN_train_test(df, use_OHE = False, to_drop = None):
    
    from lmmnn.nn_v2 import reg_nn_ohe_or_ignore, reg_nn_lmm, reg_nn_embed, reg_nn_rnn
    from lmmnn.simulation import Count
#     from lmmnn.utils import *
    import gc
    import tensorflow.keras.backend as K
    import tensorflow as tf
    
    dataframe = df.copy()
    
    if to_drop:
        dataframe = dataframe.drop(to_drop, axis=1)

    if use_OHE == True:

        column_to_encode = [col for col in dataframe.columns if col if col.startswith('g')]
        encoder = OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(dataframe[column_to_encode])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(column_to_encode))
        final_df = pd.concat([dataframe, encoded_df], axis=1)
        dataframe = final_df.copy()

        column_to_encode = [col for col in dataframe.columns if col if col.startswith('g')]
        dataframe.rename(columns={f'{col}': f'z{i}' for i, col in enumerate(column_to_encode)}, inplace=True)

    else:
        dataframe.rename(columns={[col for col in dataframe.columns if col if col.startswith('g')][0]: 'z0'}, inplace=True)

    z_cols = sorted([col for col in dataframe.columns if col.startswith('z')], key=lambda x: int(x[1:]))

    mode = 'intercepts'
    n_cats = [len(dataframe[i].unique()) for i in z_cols]
    qs = n_cats
    batch_size = 100
    epochs = 500
    patience = 10
    n_sig2bs = len(n_cats)
    n_sig2bs_spatial = 0
    est_cors = []
    n_neurons = [50,25,12,6]
    activation = 'linear'
    dropout = []
    spatial_embedded_neurons = []
    dist_matrix = None
    q_spatial = None

    print('n_uniques: ', n_cats)

    X, y = dataframe.drop('y', axis=1), dataframe['y']
    x_cols = [col for col in X.columns if col if not col.startswith('z')]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train[x_cols] = scaler.fit_transform(X_train[x_cols])
    X_test[x_cols]  = scaler.transform(X_test[x_cols])

    def reg_nn(X_train, X_test, y_train, y_test, n_cats, batch=batch_size, epochs=epochs, \
               patience=patience, reg_type='lmm', verbose=False):

        start = time.time()
        if reg_type == 'lmm':
            y_pred_tr, y_pred_fe, b_hat,history, y_pred, sigmas, _, _, n_epochs = reg_nn_lmm(X_train, X_test, y_train, y_test, n_cats, \
                                                                               q_spatial, x_cols, batch, epochs, patience,
                                                     n_neurons, dropout, activation,
                                                     mode=mode, n_sig2bs=n_sig2bs, n_sig2bs_spatial=n_sig2bs_spatial,
                                                     est_cors=est_cors, dist_matrix=dist_matrix,
                                                     spatial_embed_neurons=spatial_embedded_neurons, verbose=verbose, log_params=False)
        else:
            raise ValueError(reg_type + 'is an unknown reg_type')
        end = time.time()
        gc.collect()
        K.clear_session()
        mse = np.mean((y_pred - y_test)**2)
        return mse, sigmas, n_epochs, end - start, y_pred, y_pred_tr, y_pred_fe, b_hat

    mse_lmm, sigmas, n_epochs_lmm, time_lmm,y_pred, y_pred_tr, y_pred_fe, b_hat= reg_nn(X_train, X_test, y_train, y_test, n_cats, reg_type='lmm', verbose=False)
    
    y_train = y_train[X_train.index]
    y_pred_train = y_pred_tr + b_hat[X_train['z0']]
    
    mse_train = np.mean((y_pred_train - y_train)**2)
    mse_test = np.mean((y_test- y_pred)**2)
    
    return mse_train, mse_test, time_lmm, y_test, y_pred
