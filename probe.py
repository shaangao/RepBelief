from __future__ import print_function
import os
import random
import copy
import argparse
from pathlib import Path
import math
import pickle
import json
import glob
import time
import torch
from datetime import datetime

# Data & Computation
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, log_loss
from sklearn.metrics import roc_curve
from tqdm import tqdm

import plotly.express as px
import plotly.io as pio

import ipdb

import matplotlib.colors as colors
from sklearn.model_selection import train_test_split, GroupKFold
import seaborn as sns
from utils import *


"""
run script: 
python probe.py --belief=protagonist --dynamic=0_forward --variable belief --model_name Qwen3-8B --num_test 0 --seed 1234
python probe.py --belief=oracle --dynamic=0_forward --variable belief --model_name Qwen3-8B --num_test 0 --seed 1234

python probe.py --belief=protagonist --dynamic=0_forward --variable action --model_name Qwen3-8B --num_test 0 --seed 1234
python probe.py --belief=oracle --dynamic=0_forward --variable action --model_name Qwen3-8B --num_test 0 --seed 1234

python probe.py --belief=protagonist --dynamic=0_backward --variable belief --model_name Qwen3-8B --num_test 0 --seed 1234
python probe.py --belief=oracle --dynamic=0_backward --variable belief --model_name Qwen3-8B --num_test 0 --seed 1234
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen3-8B')
    parser.add_argument('--belief', type=str, default='protagonist')
    parser.add_argument('--dynamic', type=str, default='0_forward')
    parser.add_argument('--variable', type=str, default='belief')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num_test', type=int, default=0)
    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args


def plot_heatmap(ht, name, save_path=None, vmin=0.5, vmax=1, cmap='Greens'):
    # Increase global font size for all text elements
    plt.rcParams.update({'font.size': 22})

    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap using seaborn
    sns.heatmap(ht, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'drawedges': False}, square=True)

    # Customize the colorbar
    cbar = ax.collections[0].colorbar
    cbar.outline.set_linewidth(2)  # Set colorbar outline width

    # Set the ticks for x and y axes with specified interval
    ax.set_xticks(np.arange(0.5, ht.shape[1], 5))
    ax.set_yticks(np.arange(0.5, ht.shape[0], 5))

    # Set the tick labels for x and y axes with specified interval and keep x-axis labels horizontal
    ax.set_xticklabels(np.arange(0, ht.shape[1], 5), rotation=0)
    ax.set_yticklabels(np.arange(0, ht.shape[0], 5))

    # Set axis labels and title with increased padding and font size
    ax.set_xlabel('Head', fontsize=24, labelpad=20)
    ax.set_ylabel('Layer', fontsize=24, labelpad=20)
    # ax.set_title(name, fontsize=28)

    # Reinstate axis lines with specified linewidth
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_visible(True)
        ax.spines[axis].set_linewidth(2)

    # Optionally save the figure as a PDF with vectorized content
    if save_path:
        # plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(save_path + '.png', format='png', dpi=300, bbox_inches='tight')

    # Clear the current figure's memory to prevent resource leaks
    plt.close(fig)
    

# probe a single (layer, head) combination
def probe_single_case(X_train, y_train, X_val, y_val, seed=0, verbose=False):
    clf = LogisticRegression(random_state=seed, max_iter=1000, C=10).fit(X_train, y_train)

    # training set
    y_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)

    # validation set
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    hits = (y_val_pred == y_val).astype(int)
    
    y_val_proba = clf.predict_proba(X_val)[:, 1]  # Probability estimates for the positive class
    roc_auc = roc_auc_score(y_val, y_val_proba)
    logloss = log_loss(y_val, y_val_proba)

    if verbose:
        print("Confusion Matrix (Validation Set):")
        print(confusion_matrix(y_val, y_val_pred))
        # Classification Report
        print("\nClassification Report (Validation Set):")
        print(classification_report(y_val, y_val_pred))
        # ROC-AUC Score
        print("\nROC-AUC Score (Validation Set):", roc_auc)
        print("\nLog-Loss (Validation Set):", logloss)
    return train_acc, val_acc, roc_auc, logloss, hits, clf


# load data for probing (excluding behav/causal eval stories)
def load_data(model_name, dynamic, belief, variable="belief", num_test=0):
    N = 200
    num_test = num_test  # the first 100 stories are used for behav/causal eval.
    feats_all = []
    labels_all = []
    usable = []
    for i in range(N):
        for setting_name in ['true', 'false']:
            for belief_statement in ['true', 'false']:
                feats = np.load('data/results/representations/%s/%s_%s_%s_belief/reps_0shot_%s_%s_belief_%s_%d_attention.npy' \
                    % (model_name, dynamic, variable, setting_name, variable, setting_name, belief_statement, i))
                if belief == 'protagonist':
                    label = (belief_statement=='true')
                elif belief == 'oracle':
                    label = (setting_name==belief_statement)
                else:
                    raise NotImplementedError
                feats_all.append(feats)
                labels_all.append(label)
                usable.append(i >= num_test)
    feats_all = np.array(feats_all)
    labels_all = np.array(labels_all)
    
    usable = np.array(usable)
    all_X = feats_all[usable]
    all_y = labels_all[usable]
    return all_X, all_y


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def probe_all(all_X, all_y, test_size=0.2, groups=None, seed=0):
    num_entries, num_layers, num_heads, head_dims = all_X.shape
    data_ids = np.arange(len(all_X))

    # split data into training and validation sets
    if groups is None:
        # random split without preserving story grouping
        all_X_train, all_X_val, y_train, y_val, train_idx, test_idx = \
            train_test_split(all_X, all_y, data_ids, test_size=test_size, random_state=seed)
    else:
        # split preserving story grouping
        n_splits = int(1 / test_size)
        # gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        gkf = GroupKFold(n_splits=n_splits, shuffle=False)  # for paired t test
        folds = list(gkf.split(all_X[:, 0, 0, :], all_y, groups))
        # use the first fold only
        train_idx, test_idx = folds[0]
        all_X_train, all_X_val, y_train, y_val = \
            all_X[train_idx], all_X[test_idx], all_y[train_idx], all_y[test_idx]

    # init output arrays
    train_acc_all = np.zeros([num_layers, num_heads])
    val_acc_all = np.zeros([num_layers, num_heads])
    roc_auc_all = np.zeros([num_layers, num_heads])
    logloss_all = np.zeros([num_layers, num_heads])
    hits_all = np.full([num_layers, num_heads, len(all_y)], np.nan)
    coefs_all = np.zeros([num_layers, num_heads, head_dims])
    CoMs_all = np.zeros([num_layers, num_heads, head_dims])

    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 

            # print(layer, head)
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]
            train_acc_all[layer][head], val_acc_all[layer][head], \
                roc_auc_all[layer][head], logloss_all[layer][head], \
                    hits_all[layer][head][test_idx], clf \
                        = probe_single_case(X_train, y_train, X_val, y_val, seed)
            coefs_all[layer][head] = clf.coef_[0]

            # calculate mean 
            true_mass_mean = np.mean(X_train[y_train], axis=0)
            false_mass_mean = np.mean(X_train[y_train==False], axis=0)
            CoM_false2true = true_mass_mean - false_mass_mean
            CoMs_all[layer][head] = CoM_false2true

    return train_acc_all, val_acc_all, roc_auc_all, logloss_all, hits_all, coefs_all, CoMs_all


if __name__ == "__main__":

    os.chdir('/net/projects2/ycleong/sg/strategy-rl/RepBelief')
    print(f'Current working directory: {os.getcwd()}')

    args = parse_args()
    set_random_seed(args.seed)

    # settings
    num_test = args.num_test  # n stories excluded from probing
    val_size = 0.5  # train-val split ratio
    group_stories = True  # group stories by 4 entries/story

    # load training set only (stories >= args.num_test). args.num_test stories excluded for behav/causal eval.
    all_X, all_y = load_data(
        model_name=args.model_name,
        dynamic=args.dynamic, 
        belief=args.belief, 
        variable=args.variable,
        num_test=num_test,
        # num_test=args.num_test
    )

    # group stories by 4 entries/story
    groups = np.arange(len(all_X)) // 4 if group_stories else None
    # probe all (layer, head) combinations
    train_acc_all, val_acc_all, roc_auc_all, logloss_all, hits_all, coefs_all, CoMs_all \
        = probe_all(
            all_X, 
            all_y, 
            test_size=val_size, 
            groups=groups, 
            seed=args.seed
        )

    # Save the results
    result_dir = f'data/results/probe/{args.model_name}/total{200-num_test}_val{val_size}_group{group_stories}'
    ensure_dir(result_dir)
    plot_heatmap(
        train_acc_all, "Probe Train Acc.", 
        save_path=f"{result_dir}/{args.dynamic}_{args.variable}_{args.belief}_train_acc"
    )
    plot_heatmap(
        val_acc_all, "Probe Val Acc.", 
        save_path=f"{result_dir}/{args.dynamic}_{args.variable}_{args.belief}_val_acc"
    )
    plot_heatmap(
        roc_auc_all, "ROC AUC Val", 
        save_path=f"{result_dir}/{args.dynamic}_{args.variable}_{args.belief}_val_auc"
    )

    np.save(
        f"{result_dir}/{args.dynamic}_{args.variable}_{args.belief}_val_acc.npy", 
        val_acc_all
    )
    np.save(
        f"{result_dir}/{args.dynamic}_{args.variable}_{args.belief}_val_hits.npy", 
        hits_all
    )
    np.save(
        f"{result_dir}/{args.dynamic}_{args.variable}_{args.belief}_coef.npy",
        coefs_all
    )
    