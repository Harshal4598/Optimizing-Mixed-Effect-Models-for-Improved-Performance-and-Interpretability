import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import ast

Models = {
    'mse_Linear_Exclude_Group': 'Linear model without Group Feature',
    'mse_Linear_Include_Group': 'Linear model with Group Feature',
    'mse_linearohe': 'Linear model with One-Hot',
    'mse_mixedlm': 'MixedLM Statsmodels',
    'mse_lmmnn': 'LMMNN',
    'mse_merf': 'MERF',
    'mse_armed': 'ARMED',
}

def Model(model_name):
    if model_name in Models:
        return Models[model_name]
    else:
        return model_name
    
def plot_mixed_effects(data, feature, group, legends = False):
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

    col = str(group)

    for i in np.unique(data[col]):
        sns.regplot(x = data[data[col] == i][str(feature)], y = data[data[col] == i]['y'],\
                    label = "Group:"+str(i),scatter_kws={"alpha": 1})
    
    if legends == True:
        plt.legend(loc='best')
    plt.ylabel('Target - (y)', fontsize = 12)
    plt.xlabel(f"Feature - ({feature})", fontsize = 12)
    plt.title(f"Visualizing Data with respect to group: {group}")

def plot_slope_distribution(dataframe):
    data = dataframe.groupby(['gE'], as_index=False)[dataframe.filter(like='slope').columns].mean()

    num_plots = len(data.filter(like='slope').columns)
    cols = list(data.filter(like='slope').columns)
    
    if num_plots == 1:
        fig, ax = plt.subplots(figsize = (10,5) ,dpi=100)
        ax.plot(data.gE, data[cols[0]], marker='o',label = 'Slope_feature-0')
        ax.set_xticks(data.gE)
        ax.set_yticks([np.floor(np.min(data[cols[0]])), np.ceil(np.max(data[cols[0]]))])
        ax.set_title('Feature - f0')
    else: 
        max_cols = num_plots//2
        num_rows = (num_plots + max_cols - 1) // max_cols
        sort = np.sort((num_rows, max_cols))

        fig, ax = plt.subplots(sort[0], sort[1],figsize = (10,5) ,dpi=100)
        for i, ax in enumerate(ax.flat):
            try:
                ax.plot(data.gE, data[cols[i]], marker='o',label = 'Slope_feature-'+str(i))
                ax.set_xticks(data.gE)
                ax.set_title('Feature - f'+str(i))
            except IndexError: ax.axis('off')

    fig.supxlabel("Group Labels")
    fig.supylabel("Slope Magnitude")
    fig.suptitle("Generated Slopes")
    plt.tight_layout()
    
def plot_intercept_distribution(dataframe):
    
    fig, ax = plt.subplots(dpi=100)

    data = dataframe.groupby(['gE'], as_index=False)['intercept'].mean()

    ax.plot(data.gE, data.intercept, marker = 'o')
    ax.set_xticks(data.gE)

    fig.supxlabel("Group Labels")
    fig.supylabel("Intercept Magnitude")
    fig.suptitle("Generated Intercepts")
    plt.tight_layout()


def plot_results_range_individual_effective_group(results_df, effective_group_nr, measure, use_RMSE = False):

    grouped_data_all = results_df.groupby(['gE','gV'])[str(measure)].agg(['mean']).reset_index()
    e_num = effective_group_nr

    grouped_data = grouped_data_all[grouped_data_all.gE == e_num]
    groups = grouped_data.gV
    mean_val = grouped_data['mean']

    plt.figure(2,figsize=(10, 6))
    if use_RMSE == True:
        plt.plot(groups, np.sqrt(mean_val), marker='o', label='Effective Group = '+str(e_num))
        plt.ylabel("RMSE - "+ Model(measure), fontsize=16)
    else:        
        plt.plot(groups, mean_val, marker='o', label='Effective Group = '+str(e_num))
        plt.ylabel("MSE - "+ Model(measure), fontsize=16)
    
    plt.xlabel('Visible Groups', fontsize=16)
    plt.xticks(grouped_data.gV)
    plt.legend()

def plot_results_range_all_groups(results_df, measure, use_rmse = False):

    grouped_data_all = results_df.groupby(['gE','gV'])[str(measure)].agg(['mean']).reset_index()
    num_plots = len(grouped_data_all.gE.unique())
    e_nums = list(grouped_data_all.gE.unique())
    max_cols = num_plots//2
    num_rows = (num_plots + max_cols - 1) // max_cols
    sort = np.sort((num_rows, max_cols))

    fig, ax = plt.subplots(sort[0], sort[1],figsize=(15, 8), dpi=100, sharey=True, sharex = False)
    for i, ax in enumerate(ax.flat):
            try:
                grouped_data = grouped_data_all[grouped_data_all.gE == e_nums[i]]
                groups = grouped_data.gV
                mean_val = grouped_data['mean']

                if use_rmse == True:
                    ax.plot(groups, np.sqrt(mean_val), marker='o')
                    fig.supylabel("RMSE - "+Model(measure), fontsize=20)
                else:
                    ax.plot(groups, mean_val, marker='o')
                    fig.supylabel("MSE - "+Model(measure), fontsize=20)

                ax.legend(labels=["Effective Group="+str(int(e_nums[i]))], loc='best')

            except IndexError: ax.axis('off')

    fig.suptitle(f"Performance of {Model(measure)} on Effective Groups", fontsize=24)
    fig.supxlabel('Visible Groups', fontsize=20)
    fig.tight_layout(pad = 2)

def plot_results_3D(results_df, measure):
    
    grouped_data_all = results_df.groupby(['gE','gV'])[str(measure)].agg(['min', 'max', 'mean']).reset_index()

    fig = plt.figure(figsize=(25,10), dpi = 100)
    ax = fig.add_subplot(122,projection='3d')

    for i in np.unique(grouped_data_all.gE):

        data = grouped_data_all[grouped_data_all['gE'] == i]
        g1 = data['gE']
        g2 = data['gV']
        error = data['mean']

        ax.bar3d(g1, g2, error, dx=0.03, dy=7, dz=(-error), alpha = 0.7)
        ax.plot3D(g1, g2, error, linewidth=1.5)

    ax.set_xticks(grouped_data_all['gE'].unique().astype(int), grouped_data_all['gE'].unique().astype(int), fontsize=11)
    ax.set_yticks([10,50,100,150,200,250],[10,50,100,150,200,250], fontsize=11)
    ax.set_zticks(ax.get_zticks(),ax.get_zticks(), fontsize=11)
    ax.set_zlim([0,grouped_data_all['mean'].max()])

    ax.set_ylabel('Visible Groups', fontsize = 14,labelpad=5)
    ax.set_xlabel('Effective Groups', fontsize = 14,labelpad=3)
    ax.set_zlabel("MSE", fontsize = 14, labelpad = 2.3, rotation = 90)
    # ax.set_title(f'MSE - {Models[measure]}', fontsize=15, y=1.03)
    
def plot_ALL_model_comparision_range_individual_effective_group(results_df, use_rmse = False):
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 9), dpi=100)
    axs = axs.flatten()

    handles, labels = [], []
    measures = list(Models.keys())
    measures.insert(6, 0)
    measures.insert(8, 0)

    for p in range(1, 10):
        if p == 7 or p == 9:
            continue
        else:
            plot_number = p - 1 if p < 7 else p - 2
            data = results_df.groupby(['gE', 'gV'])[str(measures[p-1])].agg(['mean']).reset_index()
            for i in data.gE.unique():
                if use_rmse == True:
                    line, = axs[p - 1].plot(data[data.gE == i]['gV'], np.sqrt(data[data.gE == i]['mean']),\
                                           label="Effective Groups: "+str(int(i)))
                    fig.supylabel("RMSE", fontsize = 24)
                    if p == 1:
                        handles.append(line)
                        labels.append("Effective Groups: "+str(int(i)))
                else:
                    line, = axs[p - 1].plot(data[data.gE == i]['gV'], data[data.gE == i]['mean'],\
                                           label="Effective Groups: "+str(int(i)))
                    fig.supylabel("MSE")
                    if p == 1:
                        handles.append(line)
                        labels.append("Effective Groups: "+str(int(i)))
            axs[p - 1].set_xlabel("Visible Groups")
            axs[p - 1].set_title(Models[measures[p-1]])

    for index in [6, 8]:
        fig.delaxes(axs[index])

    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0, rect=[0, 0, 1, 0.95])
    fig.legend(handles, labels, loc='lower center', ncol=1, bbox_to_anchor=(0.85, 0.11))

def plot_combined_model_comparision_mean_performance_individual_effective_group(results_df, effective_group_nr, use_rmse=False):
    
    mse_modelNames = [col for col in results_df.columns if col.startswith("mse")]
        
    fig = plt.figure(figsize=(10, 5), dpi = 100)
    gs = GridSpec(1, 2, width_ratios=[5, 1])

    ax1 = fig.add_subplot(gs[0])
    for i in mse_modelNames:

        if i == 'mse_linearohe': 
            gV_min = results_df[results_df[i] > 100]['gV'].min()
            grouped_data_all = results_df[results_df.gV < gV_min].groupby(['gE','gV'])[str(i)].agg(['mean']).reset_index()
        else:
            grouped_data_all = results_df.groupby(['gE','gV'])[str(i)].agg(['mean']).reset_index()

        grouped_data = grouped_data_all[(grouped_data_all.gE == effective_group_nr)]
        groups = grouped_data.gV
        mean_val = grouped_data['mean']
        if use_rmse == True:
            ax1.plot(groups, np.sqrt(mean_val), marker='o',label=Models[str(i)])
            ax1.set_ylabel('RMSE', fontsize=14)
        else:
            ax1.plot(groups, mean_val, marker='o',label=Models[str(i)])
            ax1.set_ylabel('MSE', fontsize=14)
        ax1.set_xlabel('Visible Groups', fontsize=14)
        ax1.legend(loc = 'best')


    ax2 = fig.add_subplot(gs[1])
    target_y = ast.literal_eval(results_df[results_df['gE'] == effective_group_nr].iloc[0,-1])
    ax2.boxplot(target_y)
    ax2.set_xlabel("Target (y)", fontsize=14)
    ax2.set_xticks([],[])
    ax2.set_ylabel("Prediction Range", fontsize=14)

    fig.suptitle("Model Comparison for Effective Group: "+str(effective_group_nr), fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    
def plot_model_comparison_effective_and_visible_groups(data_effective, data_visible, effective_group_nr, use_rmse=False):
    
    res_eff = data_effective[data_effective['gE'] == effective_group_nr]
    res = data_visible[data_visible['gE'] == effective_group_nr]

    fig, axs = plt.subplots(2,2, figsize=(15, 9), dpi=100)
    axs = axs.flatten()

    measures = ['mse_mixedlm', 'mse_lmmnn', 'mse_merf', 'mse_armed']

    for itr, m in enumerate(measures):

        data_gE = res_eff.groupby(['gE','gV'])[m].agg(['mean']).reset_index()
        data_gV = res.groupby(['gE','gV'])[m].agg(['mean']).reset_index()

        repeated_rows = []
        for value in list(range(10,260,20)):
            new_row = data_gE.copy()
            new_row['gV'] = value
            repeated_rows.append(new_row)
        data_gE = pd.concat(repeated_rows, ignore_index=True)

        if use_rmse == True:
            axs[itr].plot(data_gE.gV, np.sqrt(data_gE['mean']), marker = 'o', label="Effective Groups")
            axs[itr].plot(data_gV.gV, np.sqrt(data_gV['mean']), marker = '*', label="Visible Groups")
            axs[itr].set_title(Models[m], fontsize = 16)
            fig.supylabel("RMSE", fontsize = 24)
        else:
            axs[itr].plot(data_gE.gV, data_gE['mean'], marker = 'o', label="Effective Groups")
            axs[itr].plot(data_gV.gV, data_gV['mean'], marker = 'o', label="Visible Groups")
            axs[itr].set_title(Models[m], fontsize = 16)
            fig.supylabel("MSE", fontsize = 24)

    axs[0].legend(loc = "best", fontsize = 13)
    fig.supxlabel("Visible Groups", fontsize = 24)
    fig.suptitle("Model Comparison for Effective Group: "+str(effective_group_nr), fontsize=20, y = 0.95)
    plt.tight_layout(pad=1.5, h_pad=1.0, w_pad=1.0, rect=[0, 0, 1, 0.95])



def plot_boxplot_MSE_comparision(results_df,n_effective_group, n_visible_group, measure):
    
    filtered_data = results_df[(results_df.gE == n_effective_group) & (results_df.gV == n_visible_group)]
    xtick_labels = [f"Seed: {seed}\nMSE: {mse:.3f}" for seed, mse in zip(filtered_data.seed, filtered_data[measure])]
    
    try:
        filtered_data['Target_y'] = [ast.literal_eval(i) for i in filtered_data['Target_y']]
    except ValueError: pass
    
    target_y = [i for i in filtered_data['Target_y']]

    fig, ax = plt.subplots(figsize = (10,5), dpi=100)

    plt.boxplot(target_y)
    plt.ylabel("Y", fontsize = 15)
    plt.xticks([1,2,3,4,5], xtick_labels, fontsize = 13)
    plt.text(0.5, 1.08, "Comparision of Target Y with MSE of "+ Model(measure), 
             fontsize=17, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.text(0.5, 1.005, "Effective group: "+str(n_effective_group)+"  Visible group: "+str(n_visible_group), 
             fontsize=13, ha='center', va='bottom', transform=plt.gca().transAxes)

def plot_results_seed_individual_effective_group(results_df, effective_group_nr, measure):
    
    e_num = effective_group_nr
    grouped_data = results_df[results_df.gE == e_num]
    groups = grouped_data.gV
    mean_val = grouped_data['mean']

    plt.figure(2,figsize=(10, 6))
    
    for i in np.unique(grouped_data.seed):
        data = grouped_data[grouped_data.seed == i]
        plt.plot(data.gV, data[str(measure)], marker='o', label='seed: '+str(int(i)))
    
    plt.xlabel('Visible Groups', fontsize=16)
    plt.ylabel("MSE - "+Model(measure), fontsize=16)
    plt.title(str(input("Enter Figure Name: "))+" Effective Group: "+str(int(e_num)), fontsize=20)
    plt.legend()