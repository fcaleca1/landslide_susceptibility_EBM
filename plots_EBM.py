import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mpl_toolkits.mplot3d.axes3d as p3
import math
import seaborn as sns
from interpret import show

def plot_continuos_effects(model, feature_indices, feature_labels):
    ebm_global = model.explain_global()

    num_features = len(feature_indices)
    num_cols = math.ceil(math.sqrt(num_features))
    num_rows = math.ceil(num_features / num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, dpi=300, figsize=(20, 5*num_rows))

    for i, feature_index in enumerate(feature_indices):
        row_idx = i // num_cols
        col_idx = i % num_cols

        
        bagged_data=model.bagged_scores_[feature_index]
        bagged_data=bagged_data[:, 1:-1]
        std_tot=bagged_data.std(axis=0)
        std = np.r_[std_tot, std_tot[np.newaxis, -1]]
        shape_data = ebm_global.data(feature_index)
        x_vals = shape_data["names"].copy()
        y_vals = shape_data["scores"].copy()
        y_vals = np.r_[y_vals, y_vals[np.newaxis, -1]]
        sum_=y_vals+std
        min_=y_vals-std
        x = np.array(x_vals)
        bins = np.linspace(min(x), max(x), 50)

        counts, _ = np.histogram(x, bins=bins)
        
        
        colors = counts / np.max(counts)
        colors = ["white","powderblue","lightblue","skyblue", "steelblue"]
        cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
        if row_idx == num_rows-1 and col_idx >= num_features % num_cols:
            
            axs[row_idx, col_idx].axis("off")
            continue
        else:
            ax = axs[row_idx, col_idx]
    
        ax = axs[row_idx, col_idx]
        ax.step(x, y_vals, where="post", color="palevioletred", linewidth=1.5)
        ax.fill_between(x,min_,sum_,color='gray',alpha=0.2)
        ax.set_ylabel("Effect on prediction",fontsize=18)
        ax.set_ylim(-1.5,1.5)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlim(min(x), max(x))
        ax.set_xlabel(feature_labels[i],fontsize=18)
        ax.tick_params(axis='x', labelsize=14)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)

        ax2 = ax.twinx()
        ax2.bar(
            bins[:-1], np.ones(len(counts)), width=np.diff(bins), color=cmap(counts / np.max(counts)), alpha=0.6
        )
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax.set_zorder(ax2.get_zorder()+1)
        ax.patch.set_visible(False)
        ax.patch.set_visible(False)
    for i in range(num_features, num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].axis("off")
    plt.tight_layout()
    return fig

def plot_categorical_effects(model, feature_indices, feature_labels):
    ebm_global = model.explain_global()
    num_features = len(feature_indices)
    num_cols = 2
    num_rows = 2
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, dpi=300, figsize=(20, 5*num_rows))
    for i, feature_index in enumerate(feature_indices):
        row_idx = i // num_cols
        col_idx = i % num_cols

       
        bagged_data=model.bagged_scores_[feature_index]
        bagged_data=bagged_data[:, 1:-1]
        std_tot=bagged_data.std(axis=0)
        shape_data = ebm_global.data(feature_index)
        x_vals = shape_data["names"].copy()
        y_vals = shape_data["scores"].copy()
        
        
        ax = axs[row_idx, col_idx]
        ax.errorbar(x_vals,y_vals, yerr=std_tot,mfc='blue',capsize=4,ecolor='black',fmt='o',mec='black')
        ax.set_ylabel("Effect on prediction",fontsize=18)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_ylim(-1.5,1.5)
    plt.tight_layout()
    return fig

def plot_interaction (data, column_x, column_y, feature_index, model):
    ebm_global = model.explain_global()
    bagged_data = model.bagged_scores_[feature_index]
    bagged_data = bagged_data[:, 1:-1]
    std_tot = bagged_data.std(axis=0)
    shape_data = ebm_global.data(feature_index)
    x_vals = shape_data["left_names"].copy()
    y_vals = shape_data["right_names"].copy()
    z = shape_data["scores"].copy()
    z_vals = np.transpose(z, (1, 0))
    x_new = x_vals[1:]
    y_new = y_vals[1:]
    
    sns.set(style="ticks")  
    g = sns.jointplot(x=data[column_x], y=data[column_y], kind='hist', bins=(np.arange(0.025, 0.25, 0.025), np.arange(70, 140, 10)), color='grey')
    g.ax_marg_x.tick_params(axis='x', bottom=False, labelbottom=False)
    g.ax_marg_y.tick_params(axis='y', left=False, labelleft=False)
    
    contour = g.ax_joint.contourf(x_new, y_new, z_vals, cmap='viridis', levels=7)
    g.ax_joint.set_xlim(0.025, max(x_vals))
    g.ax_joint.set_ylim(70, 130)
    g.ax_joint.set_yticks(np.arange(70, 140, 10))
    g.ax_joint.set_xticks(np.arange(0.025, 0.25, 0.025))
    g.ax_joint.set_ylabel(column_y, fontsize=16, fontfamily='calibri', fontweight='bold')
    g.ax_joint.set_xlabel(column_x, fontsize=16, fontfamily='calibri', fontweight='bold')
    g.ax_marg_x.tick_params(axis='y', left=False, labelleft=False)
    g.ax_marg_y.tick_params(axis='x', bottom=False, labelbottom=False)
    plt.xticks(np.arange(0.025, 0.25, 0.025), fontsize=16, fontfamily='calibri')
    plt.yticks(np.arange(70, 140, 10), fontsize=16, fontfamily='calibri')
    plt.tight_layout()
    return g

def local_effects (model,X, index_vl,index_m,index_vh):
    obs=len(X)+1
    explanations=model.explain_local(X[:obs], y[:obs])
    names_vl=explanations.data(index_vl)['names']
    scores_vl=explanations.data(index_vl)['scores']
    intercept=explanations.data(index_vl)['extra']['names']
    scor_interc=explanations.data(index_vl)['extra']['scores']
    names_vl=names_vl+intercept
    scores_vl=scores_vl+scor_interc
    
    names_m=explanations.data(index_m)['names']
    scores_m=explanations.data(index_m)['scores']
    names_m=names_m+intercept
    scores_m=scores_m+scor_interc
    
    names_vh=explanations.data(index_vh)['names']
    scores_vh=explanations.data(index_vh)['scores']
    names_vh=names_vh+intercept
    scores_vh=scores_vh+scor_interc
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    axs[0].barh(names_vl, scores_vl,edgecolor='black')
    axs[0].set_xlim(-1.5,1.5)
    axs[0].set_xlabel('Effect')
    axs[0].axvline(x=0, color="black", linestyle="--")

    axs[1].barh(names_m, scores_m,edgecolor='black')
    axs[1].set_xlim(-1.5,1.5)
    axs[1].set_xlabel('Effect')
    axs[1].axvline(x=0, color="black", linestyle="--")

    axs[2].barh(names_vh, scores_vh,edgecolor='black')
    axs[2].set_xlim(-1.5,1.5)
    axs[2].set_xlabel('Effect')
    axs[2].axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()
    return fig
