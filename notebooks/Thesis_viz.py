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
    'mse_mixedlm': 'MixedLM statmodels',
    'mse_lmmnn': 'LMMNN',
    'mse_merf': 'MERF',
    'mse_armed': 'ARMED',
}

def plot_mixed_effects(data, feature, group):
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

    col = str(group)

    for i in np.unique(data[col]):
        sns.regplot(data[data[col] == i][str(feature)], data[data[col] == i]['y'],label = "Group:"+str(i),scatter_kws={"alpha": 0.5})
    plt.legend(loc='best')
    plt.ylim([data.y.min()-10, data.y.max()+10])
    plt.title("Hue: "+str(group))
        
    return plt.show()

def plot_results_range_individual_effective_group(results_df, effective_group_nr, measure):
    
    grouped_data_all = results_df.groupby(['gE','gV'])[str(measure)].agg(['min', 'max', 'mean']).reset_index()
    e_num = effective_group_nr

    grouped_data = grouped_data_all[grouped_data_all.gE == e_num]
    groups = grouped_data.gV
    min_val = grouped_data['min']
    max_val = grouped_data['max']
    mean_val = grouped_data['mean']

    plt.figure(2,figsize=(10, 6))
    plt.plot(groups, mean_val, marker='o', label='Effective Group = '+str(e_num))
    plt.fill_between(groups, min_val, max_val, alpha=0.2)
    plt.xlabel('Visible Groups', fontsize=16)
    plt.ylabel("MSE - "+Models[str(measure)], fontsize=16)
    plt.title(str(input("Enter Figure Name: ")), fontsize=20)
    plt.legend()
              
    return plt.show()

def plot_results_seed_individual_effective_group(results_df, effective_group_nr, measure):
    
    e_num = effective_group_nr
    grouped_data = results_df[results_df.gE == e_num]

    plt.figure(2,figsize=(10, 6))
    
    for i in np.unique(grouped_data.seed):
        data = grouped_data[grouped_data.seed == i]
        plt.plot(data.gV, data[str(measure)], marker='o', label='seed: '+str(int(i)))
    
    plt.xlabel('Visible Groups', fontsize=16)
    plt.ylabel("MSE - "+Models[str(measure)], fontsize=16)
    plt.title(str(input("Enter Figure Name: "))+" Effective Group: "+str(int(e_num)), fontsize=20)
    plt.legend()
              
    return plt.show()

def plot_results_range_all_groups(results_df, measure):
    
    grouped_data_all = results_df.groupby(['gE','gV'])[str(measure)].agg(['min', 'max', 'mean']).reset_index()

    fig, ax = plt.subplots(3,4, figsize=(24, 12), dpi=100, sharey=True, sharex = False)
    gs = GridSpec(4, 4, figure=fig, wspace=0.0, hspace=0.0)
    
    itr = 1
    for itr1 in [0,1,2]:
        for itr2 in [0,1,2,3]:

            itr = itr + 1
            grouped_data = grouped_data_all[grouped_data_all.gE == itr]
            groups = grouped_data.gV
            min_val = grouped_data['min']
            max_val = grouped_data['max']
            mean_val = grouped_data['mean']

            ax[itr1,itr2].plot(groups, mean_val, marker='o')
            ax[itr1,itr2].fill_between(groups, min_val, max_val, alpha=0.2)
            ax[itr1,itr2].legend(labels=["Effective Group="+str(itr)], loc='best')

    fig.suptitle(str(input("Enter Figure Name: ")), fontsize=24)
    fig.supxlabel('Visible Groups', fontsize=20)
    fig.supylabel("MSE - "+Models[str(measure)], fontsize=20)
    fig.tight_layout(pad = 2)
    
    return plt.show()

def plot_results_seed_all_groups(results_df, measure):
        
    fig, ax = plt.subplots(3,4, figsize=(24, 12), dpi=100, sharey=True, sharex = False)
    gs = GridSpec(4, 4, figure=fig, wspace=0.0, hspace=0.0)
    
    itr = 1
    for itr1 in [0,1,2]:
        for itr2 in [0,1,2,3]:
            
            itr = itr + 1
            
            for i in np.unique(results_df.seed):
                
                data = results_df[(results_df.seed == i) & (results_df.gE == itr)]
                ax[itr1,itr2].plot(data.gV, data[str(measure)], marker='o',label='seed: '+str(i))
                ax[itr1,itr2].set_title("Effective Group="+str(itr))
    
    ax[0,0].legend()
    fig.suptitle(str(input("Enter Figure Name: ")), fontsize=24)
    fig.supxlabel('Visible Groups', fontsize=20)
    fig.supylabel("MSE - "+Models[str(measure)], fontsize=20)
    fig.tight_layout(pad = 2)
    
    return plt.show()

def plot_results_3D(results_df, measure):
    
    grouped_data_all = results_df.groupby(['gE','gV'])[str(measure)].agg(['min', 'max', 'mean']).reset_index()

    fig = plt.figure(figsize=(18, 10), dpi = 100)
    ax = fig.add_subplot(122,projection='3d')

    for i in np.unique(grouped_data_all.gE):

        data = grouped_data_all[grouped_data_all['gE'] == i]
        g1 = data['gE']
        g2 = data['gV']
        error = data['mean']

        ax.bar3d(g1, g2, error, dx=0.03, dy=7, dz=(-error), alpha = 0.7)
        ax.plot3D(g1, g2, error, linewidth=1.5)

    ax.set_zlim([0,grouped_data_all['mean'].max()])
    ax.set_xticks(grouped_data_all['gE'].unique())

    ax.set_ylabel('Visible Groups')
    ax.set_xlabel('Effective Groups')
    ax.set_zlabel("MSE - "+Models[str(measure)])
    ax.set_title(str(input("Enter Figure Name: ")), fontsize=15, y=1.03)
    plt.tight_layout()
    
    return plt.show()

def plot_model_comparision_mean_performance(results_df):
    
    mse_modelNames = [col for col in results_df.columns if col.startswith("mse")]
        
    fig, ax = plt.subplots(3,4, figsize=(24, 12), dpi=100, sharey=True, sharex = False)
    gs = GridSpec(4, 4, figure=fig, wspace=0.0, hspace=0.0)
    
    itr = 1
    for itr1 in [0,1,2]:
        for itr2 in [0,1,2,3]:

            itr = itr + 1

            for i in mse_modelNames:
                
                if i == 'mse_linearohe': 
                    gV_min = results_df[results_df[i] > 100]['gV'].min()
                    grouped_data_all = results_df[results_df.gV < gV_min].groupby(['gE','gV'])[str(i)].agg(['min', 'max', 
                                                                                                            'mean']).reset_index()
                else:
                    grouped_data_all = results_df.groupby(['gE','gV'])[str(i)].agg(['min', 'max', 'mean']).reset_index()
                
                grouped_data = grouped_data_all[(grouped_data_all.gE == itr)]
                groups = grouped_data.gV
                mean_val = grouped_data['mean']
                
                ax[itr1,itr2].plot(groups, np.log(mean_val), marker='o',label=str(i))
                ax[itr1,itr2].set_title("Effective Group="+str(itr))
    
    ax[0,0].legend(loc = 'upper left')
    fig.suptitle(str(input("Enter Figure Name: ")), fontsize=24)
    fig.supxlabel('Visible Groups', fontsize=20)
    fig.supylabel('Log(MSE)', fontsize=20)
    fig.tight_layout(pad = 2.2)
    
    return plt.show()
    
def plot_slope_distribution(dataframe):
    
    fig, ax = plt.subplots(dpi=100)

    data = dataframe.groupby(['gE'], as_index=False)['slope'].mean()

    sns.lineplot(data.gE, data.slope, marker = 'o')
    plt.xlabel("Effective Group")
    plt.xticks(data.gE)
    plt.ylabel("Slope")
    plt.title("Slopes Distribution")
    return plt.show()

def plot_intercept_distribution(dataframe):
    
    fig, ax = plt.subplots(dpi=100)

    data = dataframe.groupby(['gE'], as_index=False)['intercept'].mean()

    sns.lineplot(data.gE, data.intercept, marker = 'o')
    plt.xlabel("Effective Group")
    plt.xticks(data.gE)
    plt.ylabel("Intercept")
    plt.title("Intercepts Distribution")
    return plt.show()

def plot_intercept_slope_distribution(dataframe):
    
    fig, ax = plt.subplots(1,2,figsize=(10,5),dpi=100)

    data = dataframe.groupby(['gE'], as_index=False)['intercept','slope'].mean()

    ax[0].plot(data.gE, data.intercept, marker = 'o')
    ax[0].set_xlabel("Effective Group")
    ax[0].set_xticks(data.gE)
    ax[0].set_ylabel("Intercept")
    ax[0].set_title("Intercepts Distribution")

    ax[1].plot(data.gE, data.slope, marker = 'o')
    ax[1].set_xlabel("Effective Group")
    ax[1].set_xticks(data.gE)
    ax[1].set_ylabel("Slope")
    ax[1].set_title("Slopes Distribution")

    plt.subplots_adjust(wspace=0.5)

    return plt.show()

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
    plt.text(0.5, 1.08, "Comparision of Target Y with MSE of "+ Models[str(measure)], 
             fontsize=17, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.text(0.5, 1.005, "Effective group: "+str(n_effective_group)+"  Visible group: "+str(n_visible_group), 
             fontsize=13, ha='center', va='bottom', transform=plt.gca().transAxes)

    plt.show()



