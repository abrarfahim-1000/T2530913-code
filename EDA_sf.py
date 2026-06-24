#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_extract_features(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            r = json.loads(line)

            rho = np.array(r.get('rho', []))
            load = np.array(r.get('load_p', []))
            gen = np.array(r.get('gen_p', []))

            if len(rho) == 0: continue

            # Forcing strict python scalars (float/int/str) to prevent Pandas hashing errors
            features = {
                'label': str(r['label']),
                'max_rho': float(np.max(rho)),
                'mean_rho': float(np.mean(rho)),
                'std_rho': float(np.std(rho)),
                'rho_above_90_pct': int(np.sum(rho > 0.90)), 
                'rho_above_100_pct': int(np.sum(rho >= 1.0)), 
                'total_load': float(np.sum(load)),
                'total_gen': float(np.sum(gen)),
                'fault_loc': int(r['fault_loc']) if r.get('fault_loc') is not None else -1
            }
            data.append(features)

    # Explicitly ensure we return a Pandas DataFrame, not a dict or list
    df = pd.DataFrame(data)
    return df
df = load_and_extract_features("data/grid_dataset_neurips2020.jsonl")


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set global seaborn style for academic-looking plots
sns.set_theme(style="whitegrid", palette="muted")

def plot_class_distribution(df):
    """
    Visualizes the severe class imbalance in the dataset.
    """
    plt.figure(figsize=(10, 6))

    # Calculate counts and percentages
    counts = df['label'].value_counts()
    percentages = (counts / len(df)) * 100

    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False)

    plt.title('Distribution of Grid States (Class Imbalance)', fontsize=14, pad=15)
    plt.xlabel('Grid State (Label)', fontsize=12)
    plt.ylabel('Number of Snapshots', fontsize=12)

    # Add percentage annotations on top of the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{percentages.iloc[i]:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, xytext=(0, 5), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()


# In[3]:


def plot_max_rho_by_class(df):
    """
    Shows the distribution of the maximum line loading percentage (max_rho) across different states.
    A reference line is drawn at 1.0 (100% capacity).
    """
    plt.figure(figsize=(10, 6))

    sns.boxplot(data=df, x='label', y='max_rho', hue='label', legend=False, showfliers=False)

    # Add a red dashed line representing the critical thermal threshold (100%)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='100% Thermal Limit')

    plt.title('Maximum Line Loading (Rho) by Grid State', fontsize=14, pad=15)
    plt.xlabel('Grid State (Label)', fontsize=12)
    plt.ylabel('Max Rho (Line Loading Proportion)', fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[4]:


def plot_generation_vs_load(df):
    """
    A scatter plot to observe if extreme load/generation imbalances correlate with specific grid failures.
    """
    plt.figure(figsize=(10, 6))

    sns.scatterplot(data=df, x='total_load', y='total_gen', hue='label', alpha=0.7, palette="deep")

    plt.title('Total Active Generation vs. Total Active Load', fontsize=14, pad=15)
    plt.xlabel('Total Active Load (MW)', fontsize=12)
    plt.ylabel('Total Active Generation (MW)', fontsize=12)

    # Place legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


# In[5]:


def plot_critical_lines_histogram(df):
    """
    Histograms showing the frequency of states with lines exceeding 90% and 100% capacity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Filter out normal states to focus on stressed grids
    stressed_df = df[df['label'] != 'normal']

    sns.histplot(data=stressed_df, x='rho_above_90_pct', bins=15, ax=axes[0], color='orange', kde=True)
    axes[0].set_title('Frequency of Lines > 90% Capacity (Stressed States)', fontsize=12)
    axes[0].set_xlabel('Number of Lines')
    axes[0].set_ylabel('Frequency')

    sns.histplot(data=stressed_df, x='rho_above_100_pct', bins=15, ax=axes[1], color='red', kde=True)
    axes[1].set_title('Frequency of Lines >= 100% Capacity (Stressed States)', fontsize=12)
    axes[1].set_xlabel('Number of Lines')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# In[6]:


# --- Execution Cell ---
# Run these after loading your 'df' using your provided function
plot_class_distribution(df)
plot_max_rho_by_class(df)
plot_generation_vs_load(df)
plot_critical_lines_histogram(df)

