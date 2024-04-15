import numpy as np
import pandas as pd
from collections import Counter
import random


def get_effective_visible_splits(size, eff, visi):

    def get_splits(size, num_splits):

        base_split_size, remainder = divmod(size, num_splits)
        splits = [base_split_size] * num_splits
        for i in range(remainder):
            splits[i] += 1

        return splits

    eff_splits = get_splits(size, eff)
    inter_splits = get_splits(visi, eff)

    visi_splits = []
    for (i_size, i_visi)  in zip(eff_splits, inter_splits):
        visi_splits.append(get_splits(i_size, i_visi))

    visi_splits = [item for sublist in visi_splits for item in sublist]

    return eff_splits, visi_splits

def create_data(n, 
                n_effective_groups, 
                n_visible_groups, 
                n_contineous_features, 
                mode, 
                fixed_slope = (0,1), 
                fixed_intercept = 0,
                fixed_error = (0,1),
                random_effects_distribution = "linspace", # or "normal"
                re_intercept = (-10,10),
                re_slope = (-10,10), 
                error_type = "on_groups", # or "on_target"
                re_error = (0,3),
                random_seed = 100,
                shuffle_groups = False,
                show_random_effects = False):

    """
    Synthetic mixed effects data generation function.

    Parameters:
    - n (int): Number of observations to generate.
    - n_effective_groups (int): Number of Effective groups
    - n_visible_groups (int): Number of Visible groups
    - n_contineous_features (int): Number of continuous features.
    - mode (str): Specifies the type of random effects to incorporate. 'intercepts', 'slopes', or 'both'.
    - fixed_slope (tuple): The fixed slope applied to all observations; defined as a Normal Distribution N(mean, std).
                            By default N(0,1).
    - fixed_intercept (float): The fixed intercept applied to all observations. By default 0.
    - fixed_error (tuple): The error range for fixed effects, defined as a Normal Distribution N(mean, std).
                            By default N(0,1).
    - random_effects_distribution (str): The distribution type of the random effects; 'linspace' or 'normal'.
                            By default 'linspace'.
    - re_intercept (tuple): The magnitude of the random intercepts applied across effective groups. 
                            If 'linspace' distribution then takes range(min, max).
                            If 'normal' distribution then takes N(mean, std).
                            By default for 'linspace': (-10,10), for 'normal' = N(0,5) 
    - re_slope (tuple): The magnitude of the random slopes applied across effective groups. 
                            If 'linspace' distribution then takes range(min, max).
                            If 'normal' distribution then takes N(mean, std).
                            By default for 'linspace': (-10,10), for 'normal' = N(0,5)
    - error_type (str): Specifies the error type based on 'on_groups' or 'on_target'.
                            If 'on_group' then applies to every visible group.
                            If 'on_target' then applies to every observation.
    - re_error (tuple): Random effects errors, applied based on the error_type. Normal distribution with N(mean, std).
                            By default N(0,3)
    - random_seed (int): Random seed value. By default 100.
    - shuffle_groups (bool): If True, groups are shuffled randomly. By default False.
    - show_random_effects (bool): If True, the function adds random effects values to the dataframe. By default False.

    Returns:
    - DataFrame: A DataFrame containing the mixed-effects data
    - Fixed slopes: The Fixed slopes of data features. 

    """
    
    size = n
    nEg = n_effective_groups
    nVg = n_visible_groups
    
    allowed_modes = ['intercept', 'slope', 'both']
    allowed_distributions = ['linspace', 'normal']
    allowed_errors = ['on_group', 'on_target']
    if mode not in allowed_modes:
        raise ValueError(f"Invalid mode '{mode}'. Please choose one of {', '.join(allowed_modes)}.")
    if random_effects_distribution not in allowed_distributions:
        raise ValueError(f"Invalid distribution '{random_effects_distribution}'. Please choose one of {', '.join(allowed_distributions)}.")
    if error_type not in allowed_errors:
        raise ValueError(f"Invalid error type '{error_type}'. Please choose one of {', '.join(allowed_errors)}.")

    cat_length = size
    np.random.seed(random_seed)

    X = np.random.uniform(-1, 1, size * n_contineous_features).reshape((size, n_contineous_features))
    betas = np.random.normal(fixed_slope[0],fixed_slope[1], size = n_contineous_features)
    data = pd.DataFrame(X)
    x_cols = ['f' + str(i) for i in range(n_contineous_features)]
    data.columns = x_cols
    e = np.random.normal(fixed_error[0], fixed_error[1], size)
    y = fixed_intercept + X @ betas + e
    
    random_effects = {}
        
    e_splits, v_splits = get_effective_visible_splits(size, nEg, nVg)
    
    if shuffle_groups == True:
        nEg_labels = list(range(nEg))
        np.random.shuffle(nEg_labels)
        data['gE'] = np.repeat(nEg_labels, e_splits)
        
        nVg_labels = list(range(nVg))
        np.random.shuffle(nVg_labels)
        data['gV'] = np.repeat(nVg_labels, v_splits)
    else:
        data['gE'] = np.repeat(range(nEg), e_splits)
        data['gV'] = np.repeat(range(nVg), v_splits)

    np.random.seed(random_seed)
    if mode == 'intercept':
        
        if random_effects_distribution == 'linspace':
            random_intercept = np.linspace(re_intercept[0], re_intercept[1], num = len(e_splits)) + \
                               np.random.normal(0,1, size = len(e_splits))
        if random_effects_distribution == 'normal':
            random_intercept = np.random.normal(re_intercept[0], re_intercept[1],size = len(e_splits))
            
        u0 = np.repeat(random_intercept, e_splits)
        random_effects['intercept'] = u0
        random_effects['Y_Fixed'] = y
        y = y + u0
        
    if mode == 'slope':
            
        X_split = np.split(X, np.cumsum(e_splits)[:-1])
        
        if random_effects_distribution == 'linspace':
            random_slope = np.linspace(re_slope[0], re_slope[1], num = len(e_splits)*n_contineous_features) + \
                     np.random.normal(0,1, size = n_contineous_features*len(e_splits))
        if random_effects_distribution == 'normal':
            random_slope = np.random.normal(re_slope[0], re_slope[1], size = len(e_splits)*n_contineous_features)

        slopes = random_slope.reshape(n_effective_groups, n_contineous_features)
        
        re_slopes = np.repeat(slopes, e_splits, axis=0)
        for i in range(re_slopes.shape[1]):
            random_effects["slope_f"+str(i)] = re_slopes[:,i] 
            
        random_effects['Y_Fixed'] = y
        
        for i in range(len(X_split)):
            X_split[i] = X_split[i] * slopes[i]
        u1_X = np.concatenate(X_split, axis=0)
        u1_X = np.sum(u1_X, axis=1)
        y = y + u1_X
        
    
    np.random.seed(random_seed)
    if mode == 'both':
        
        if random_effects_distribution == 'linspace':
            random_intercept = np.linspace(re_intercept[0], re_intercept[1], num = len(e_splits)) + \
                               np.random.normal(0,1, size = len(e_splits))
            random_slope = np.linspace(re_slope[0], re_slope[1], num = len(e_splits)*n_contineous_features) + \
                     np.random.normal(0,1, size = n_contineous_features*len(e_splits))
            
        if random_effects_distribution == 'normal':
            random_intercept = np.random.normal(re_intercept[0], re_intercept[1],size = len(e_splits))
            random_slope = np.random.normal(re_slope[0], re_slope[1], size = len(e_splits)*n_contineous_features)
        
        u0 = np.repeat(random_intercept, e_splits)
        random_effects['intercept'] = u0
        
        X_split = np.split(X, np.cumsum(e_splits)[:-1])
        slopes = random_slope.reshape(n_effective_groups, n_contineous_features)
        re_slopes = np.repeat(slopes, e_splits, axis=0)
        for i in range(re_slopes.shape[1]):
            random_effects["slope_f"+str(i)] = re_slopes[:,i] 
            
        random_effects['Y_Fixed'] = y
        
        for i in range(len(X_split)):
            X_split[i] = X_split[i] * slopes[i]
        u1_X = np.concatenate(X_split, axis=0)
        u1_X = np.sum(u1_X, axis=1)
        
        random_effects['Y_Fixed'] = y
        
        y = y + u0 + u1_X
        
    if error_type == 'on_target':
        np.random.seed(random_seed)
        y = y + np.random.normal(re_error[0], re_error[1], size=size)
    
    if error_type == 'on_groups':
        np.random.seed(random_seed)
        visible_groups_error = np.repeat(np.random.normal(re_error[0], re_error[1], size = nVg), v_splits, axis=0)
        y = y + visible_groups_error
        
    if show_random_effects == True:
        
        for effect_type, effect_values in random_effects.items():
            data[effect_type] = effect_values

    data['y'] = y
    
    return data, betas