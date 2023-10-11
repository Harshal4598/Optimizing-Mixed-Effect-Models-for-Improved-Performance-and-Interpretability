def create_data(n, n_effective_groups, n_visible_groups, n_contineous_features, mode):

    size = n
    nEg = n_effective_groups
    nVg = n_visible_groups
    
    import numpy as np
    import pandas as pd
    from collections import Counter
    import random
    
    def make_visible_split(N, m):
        base_split_size, remainder = divmod(N, m)
        splits = [base_split_size] * m  # Create splits with equal base sizes
        for i in range(remainder):
            splits[i] += 1 

        result_dict = {i: splits[i] for i in range(m)}
        return splits #result_dict
    
    def make_effective_split(arr, g):
        if g <= 0:
            raise ValueError("The value of 'g' must be greater than 0")

        combined_splits = [0] * g  # Initialize the combined_splits list with g zeros
        groups = [[] for _ in range(g)]  # Create g empty groups

        for i, num in enumerate(arr):
            group_index = i % g  # Determine the group index using modulus
            groups[group_index].append(num)  # Assign the element to the corresponding group

        for i in range(g):
            combined_splits[i] = sum(groups[i])  # Calculate the sum for each group
        
        groups = [item for sublist in groups for item in sublist]
        return combined_splits, groups

    cat_length = size # assign initial size to a category
    
    # generating fixed part simply y = mx + c
    X = np.random.uniform(-1, 1, size * n_contineous_features).reshape((size, n_contineous_features))
#     betas = np.ones(n_contineous_features)
    betas = np.random.randint(10, size = n_contineous_features)
    Xbeta = 0 + X @ betas # initial intercept 0
    fX = Xbeta
    data = pd.DataFrame(X)
    x_cols = ['f' + str(i) for i in range(n_contineous_features)]
    data.columns = x_cols
    e = np.random.normal(0, 1, size)
    y = fX + e
    
    ########### Create effective and visible splits ############
    
    v_splits = make_visible_split(size, nVg)
    e_splits, v_splits = make_effective_split(v_splits, nEg)
    
    data['gE'] = np.repeat(range(nEg), e_splits)
    random_intercept = np.random.randint(20, size = len(e_splits))
#     random_intercept = np.random.normal(0,1,size = len(e_splits))
    ue = np.repeat(random_intercept, e_splits)
    
    X_split = np.split(X, np.cumsum(e_splits)[:-1])
    slopes = np.random.uniform(-10, 10, size = len(e_splits))
    for i in range(len(X_split)):
        X_split[i] = X_split[i] * slopes[i]
    u1_X = np.concatenate(X_split, axis=0)
    u1_X = np.sum(u1_X, axis=1)
    
    if mode == 'intercept':
        y = y + ue
    if mode == 'slope':
        y = y + u1_X
    if mode == 'both':
        y = y + ue + u1_X
    
    ########### Visible Group split ############
    
    data['gV'] = np.repeat(range(nVg), v_splits)

    data['y'] = y
    return data, e_splits, v_splits

def create_intercept_data(n, n_effective_groups, n_visible_groups, n_contineous_features):

    size = n
    nEg = n_effective_groups
    nVg = n_visible_groups
    
    import numpy as np
    import pandas as pd
    from collections import Counter
    import random
    
    def make_visible_split(N, m):
        base_split_size, remainder = divmod(N, m)
        splits = [base_split_size] * m
        for i in range(remainder):
            splits[i] += 1 

        result_dict = {i: splits[i] for i in range(m)}
        return splits #result_dict
    
    def make_effective_split(arr, g):
        if g <= 0:
            raise ValueError("The value of 'g' must be greater than 0")

        combined_splits = [0] * g
        groups = [[] for _ in range(g)]

        for i, num in enumerate(arr):
            group_index = i % g
            groups[group_index].append(num)

        for i in range(g):
            combined_splits[i] = sum(groups[i])
        
        groups = [item for sublist in groups for item in sublist]
        return combined_splits, groups

    cat_length = size # assign initial size to a category
    
    # generating fixed part simply y = mx + c
    X = np.random.uniform(-1, 1, size * n_contineous_features).reshape((size, n_contineous_features))
#     betas = np.ones(n_contineous_features)
    betas = np.random.randint(10, size = n_contineous_features)
    Xbeta = 0 + X @ betas # initial intercept 0
    fX = Xbeta
    data = pd.DataFrame(X)
    x_cols = ['f' + str(i) for i in range(n_contineous_features)]
    data.columns = x_cols
    e = np.random.normal(0, 1, size)
    y = fX + e
#     data['y'] = y
    ########### Create effective and visible splits ############
    
    v_splits = make_visible_split(size, nVg)
    e_splits, v_splits = make_effective_split(v_splits, nEg)
    
    data['gE'] = np.repeat(range(nEg), e_splits)
    random_intercept = np.random.randint(20, size = len(e_splits))
#     random_intercept = np.random.normal(0,1,size = len(e_splits))
    ue = np.repeat(random_intercept, e_splits)
    y = y + ue
#     data['uE'] = ue
    
    ########### Visible Group split ############
    
    data['gV'] = np.repeat(range(nVg), v_splits)
    random_intercept = np.random.normal(-1,1,size = len(v_splits))
    uv = np.repeat(random_intercept, v_splits)
#     y = y+uv


    data['y'] = y

    return data,e_splits, v_splits