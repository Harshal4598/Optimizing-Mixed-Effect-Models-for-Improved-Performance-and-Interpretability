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

from lmmnn.nn import reg_nn_ohe_or_ignore, reg_nn_lmm, reg_nn_embed, reg_nn_rnn
from lmmnn.simulation import Count
from lmmnn.utils import *
import gc
import tensorflow.keras.backend as K

import tensorflow as tf

import logging
logging.disable(logging.INFO)
from merf import MERF


import warnings
warnings.filterwarnings('ignore')
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def Linear_Exclude_Group(df):
    
    column_to_select = [i for i in df.columns if i.startswith('f')] + ["y"]
    final_df = df[column_to_select]
    
    start = time.time()
    X,y = final_df.drop('y', axis=1), final_df.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    end = time.time()
    mse_Linear_Exclude_Group = np.mean((y_pred - y_test)**2)
    
    return mse_Linear_Exclude_Group, end-start, y_pred, y_test

def Linear_Include_Group(df):
    
    final_df = df.copy()
    
    start = time.time()
    X,y = final_df.drop('y', axis=1), final_df.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    end = time.time()
    mse_Linear_Include_Group = np.mean((y_pred - y_test)**2)
    
    return mse_Linear_Include_Group, end-start, y_pred, y_test

def LinearOHE(df):
    
    column_to_encode = ['gV']
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(df[column_to_encode])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(column_to_encode))
    final_df = pd.concat([df, encoded_df], axis=1)
    final_df.drop(['gV'], axis=1, inplace=True)
    df = final_df.copy()

    start = time.time()
    X,y = final_df.drop('y', axis=1), final_df.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end = time.time()
    mse_linearohe = np.mean((y_pred - y_test)**2)
    return mse_linearohe, end-start, y_pred, y_test

def MixedLM(df, mode):
    
    allowed_modes = ['intercept', 'slope', 'both']
    
    if mode not in allowed_modes:
        raise ValueError(f"Invalid mode '{mode}'. Please choose one of {', '.join(allowed_modes)}.")

    X, y = df.drop('y', axis=1), df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    data_train = pd.concat([X_train, y_train], axis=1)
    
    start = time.time()
    
    if mode == 'intercept':
        md = sm.MixedLM.from_formula(f"y ~ {'+'.join([i for i in df.columns if i.startswith('f')])} + (1 | gV)", data_train, groups = data_train['gV'])
    if mode == 'slope':
        md = sm.MixedLM.from_formula(f"y ~ {' + '.join([i + '*gV' for i in df.columns if i.startswith('f')])} + (1|gV)", data_train, re_formula = '~'+'+'.join([i for i in df.columns if i.startswith('f')]), groups = data_train['gV'])
    if mode == 'both':
        md = sm.MixedLM.from_formula(f"y ~ 1 + {' + '.join([i + '*gV' for i in df.columns if i.startswith('f')])} + (1|gV)", data_train, re_formula = '~1+'+'+'.join([i for i in df.columns if i.startswith('f')]), groups = data_train['gV'])
    
    mdf = md.fit()
    mlm_pred = mdf.predict(X_test)
    
    end = time.time()
    
    MSE = np.mean((mlm_pred - y_test)**2)
    
    return MSE, end-start, mlm_pred, y_test


def LMMNN(df, use_OHE = False):
    
    lmmnn_df = df.copy()
    
    if use_OHE == True:
        g_cols = [col for col in lmmnn_df.columns if col if col.startswith('g')]
        lmmnn_df['g_cart'] = lmmnn_df[g_cols].apply(lambda row: '_'.join(row.astype(str)), axis=1)
        lmmnn_df["g_cart"] = pd.factorize(lmmnn_df['g_cart'])[0]
        
        lmmnn_df.rename(columns={'g_cart': 'z0'}, inplace=True)
        lmmnn_df.rename(columns={f'{col}': f'z{i+1}' for i, col in enumerate(g_cols)}, inplace=True)
    else:
        lmmnn_df.rename(columns={'g_cart': 'z0'}, inplace=True)
        lmmnn_df.rename(columns={'gV': 'z0', 'gE': 'z1'}, inplace=True)

    z_cols = sorted([col for col in lmmnn_df.columns if col.startswith('z')], key=lambda x: int(x[1:]))

    # mode = 'slopes'
    mode = 'intercepts'
    n_cats = [len(lmmnn_df[i].unique()) for i in z_cols]
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

    X, y = lmmnn_df.drop('y', axis=1), lmmnn_df['y']
    x_cols = [col for col in X.columns if col if not col.startswith('z')]#['z0','z1','z2','z3']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train[x_cols] = scaler.fit_transform(X_train[x_cols])
    X_test[x_cols]  = scaler.transform(X_test[x_cols])

    def reg_nn(X_train, X_test, y_train, y_test, n_cats, batch=batch_size, epochs=epochs, patience=patience, reg_type='lmm', verbose=False):

        start = time.time()
        if reg_type == 'lmm':
            model_fitted, history, y_pred, sigmas, _, _, n_epochs = reg_nn_lmm(X_train, X_test, y_train, y_test, n_cats, q_spatial, x_cols, batch, epochs, patience,
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

    return mse_lmm, time_lmm, y_pred, y_test

def MERForest(df, max_iterations = 20,threshold = None):
            
    start = time.time()
    RS = 42
    test_ratio = 0.1
    val_ratio = 0.1

    X,y = df.drop('y', axis=1), df.y

    test_indices = X.sample(frac=0.8, random_state=RS).index
    split = [(np.array(list(set(X.index).difference(test_indices))), np.array(test_indices))]

    for num, (train_indices, test_indices) in enumerate(split):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RS)

    np.random.seed(100)
    clusters_train = X_train['gV']
    Z_train = np.ones(shape=(X_train.shape[0],1))
    clusters_val = X_val['gV']
    Z_val = np.ones(shape=(X_val.shape[0],1))
    
    start = time.time()
    merf = MERF(max_iterations = max_iterations, gll_early_stop_threshold=threshold)
    merf.fit(X_train, Z_train, clusters_train, y_train, X_val, Z_val, clusters_val, y_val)

    clusters_test = X_test['gV']
    X_test_merf = X_test.drop(['gV'],axis=1)
    Z_test = np.ones(shape=(X_test.shape[0],1))
    y_pred = merf.predict(X_test, Z_test, clusters_test)
    end = time.time()
    
    mse_merf = np.mean((y_pred - y_test)**2)
    
    return mse_merf, end-start, y_pred, y_test

def ARMED(df, epochs = 500):
    
    from armed.models.mlp_classifiers_reg import MixedEffectsMLP

    RS = 42
    test_ratio = 0.1
    val_ratio = 0.1
    np.random.seed(100)

    X,y = df.drop('y', axis=1), df.y

    test_indices = X.sample(frac=0.8, random_state=RS).index
    split = [(np.array(list(set(X.index).difference(test_indices))), np.array(test_indices))]

    for num, (train_indices, test_indices) in enumerate(split):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RS)

    Z_train = X_train["gV"]
    Z_val = X_val["gV"]
    Z_test = X_test["gV"]

    X_train.drop("gV",axis=1,inplace=True)
    X_val.drop("gV",axis=1,inplace=True)
    X_test.drop("gV",axis=1,inplace=True)

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