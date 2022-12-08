#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------------------
# This script plots the results from the CFA-removal in terms of ERPs for stimulus-locked
# epochs.
# -----------------------------------------------------------------------------------------


# Imports
import os
import sys
import numpy as np
import joblib
from scipy.io import loadmat
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Paths
path_predicted_data = "add_path_here"
path_file = "add_path_here"
path_meta = "add_path_here"
path_results_specificity = "add_path_here"

# insert path to color module
sys.path.insert(1, path_file)
from cool_colormaps import cga_p3_light as ccm

# Load channel XY-coordinates
channel_coords = loadmat(os.path.join(path_meta, "channel_coords.mat"))[
    "channel_coords"
][:, 0:2]

# Subject list
subject_list = [
    "VP02",
    "VP03",
    "VP04",
    "VP05",
    "VP06",
    "VP07",
    "VP08",
    "VP09",
    "VP10",
    "VP11",
    "VP12",
    "VP13",
    "VP14",
    "VP15",
    "VP16",
    "VP17",
    "VP18",
    "VP19",
    "VP20",
    "VP21",
    "VP22",
    "VP23",
    "VP24",
    "VP25",
    "VP26",
    "VP27",
    "VP28",
    "VP29",
    "VP30",
    "VP31",
    "VP32",
    "VP33",
    "VP34",
    "VP35",
    "VP36",
    "VP37",
    "VP38",
    "VP39",
    "VP40",
    "VP41",
]

# Channel list
channel_list = range(1, 61)

# Channel names
channel_names = str(
    loadmat(os.path.join(path_meta, "channel_names.mat"))["channel_names"]
)[3:-2].split(" ")

# Load erp times
erp_times_stimlock = (
    loadmat(os.path.join(path_meta, "time_stimlock.mat"))["time_stimlock"]
    .astype("float64")
    .ravel()
)

# Create a basic mne info structure
sfreq = 1000
info = mne.create_info(channel_names, sfreq, ch_types="eeg", verbose=None)

# Create a montage
standard_1020_montage = mne.channels.make_standard_montage("standard_1020")

# Lists for eveoked objects
evokeds_sys_before = []
evokeds_sys_predicetd = []
evokeds_sys_cleaned = []
evokeds_dia_before = []
evokeds_dia_predicetd = []
evokeds_dia_cleaned = []

# Iterate subjects nad create a (n_epochs, n_channels, n_times) array
for s, subject in enumerate(subject_list):

    print(f"\nProcessing {subject}...\n")

    # Load trialinfo
    trialinfo = loadmat(os.path.join(path_meta, f"{subject}_tinf.mat"))["tinf"].astype(
        "float64"
    )

    # Get number of trials
    n_epochs_sys = sum(trialinfo[:, 0] == 1)
    n_epochs_dia = sum(trialinfo[:, 0] == 2)

    # Init subject epoch matrices
    epoch_matrix_sys_before = np.zeros(
        (n_epochs_sys, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_sys_predicted = np.zeros(
        (n_epochs_sys, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_sys_after = np.zeros(
        (n_epochs_sys, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_dia_before = np.zeros(
        (n_epochs_dia, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_dia_predicted = np.zeros(
        (n_epochs_dia, len(channel_list), len(erp_times_stimlock))
    )
    epoch_matrix_dia_after = np.zeros(
        (n_epochs_dia, len(channel_list), len(erp_times_stimlock))
    )

    # iterate channels
    for c, channel in enumerate(channel_list):

        # Load stimlocked before
        dat = joblib.load(
            os.path.join(
                path_predicted_data,
                f"stim_locked_y_before_{subject}_channel_{str(channel)}.joblib",
            ),
        )
        epoch_matrix_sys_before[:, c, :] = dat[trialinfo[:, 0] == 1, :] * 1e-6
        epoch_matrix_dia_before[:, c, :] = dat[trialinfo[:, 0] == 2, :] * 1e-6

        # Load stimlocked predicted
        dat = joblib.load(
            os.path.join(
                path_predicted_data,
                f"stim_locked_y_predicted_{subject}_channel_{str(channel)}.joblib",
            ),
        )
        epoch_matrix_sys_predicted[:, c, :] = dat[trialinfo[:, 0] == 1, :] * 1e-6
        epoch_matrix_dia_predicted[:, c, :] = dat[trialinfo[:, 0] == 2, :] * 1e-6

    # Substract predictions
    epoch_matrix_sys_cleaned = epoch_matrix_sys_before - epoch_matrix_sys_predicted
    epoch_matrix_dia_cleaned = epoch_matrix_dia_before - epoch_matrix_dia_predicted

    # Create mne epochs objects
    baseline = (-0.2, 0)
    tmin = erp_times_stimlock[0] / sfreq

    epochs_sys_before = mne.EpochsArray(
        epoch_matrix_sys_before,
        info,
        tmin=tmin,
        baseline=baseline,
    )
    epochs_sys_before.set_montage(standard_1020_montage)
    epochs_sys_predicted = mne.EpochsArray(
        epoch_matrix_sys_predicted,
        info,
        tmin=tmin,
        baseline=baseline,
    )
    epochs_sys_predicted.set_montage(standard_1020_montage)
    epochs_sys_cleaned = mne.EpochsArray(
        epoch_matrix_sys_cleaned,
        info,
        tmin=tmin,
        baseline=baseline,
    )
    epochs_sys_cleaned.set_montage(standard_1020_montage)
    epochs_dia_before = mne.EpochsArray(
        epoch_matrix_dia_before,
        info,
        tmin=tmin,
        baseline=baseline,
    )
    epochs_dia_before.set_montage(standard_1020_montage)
    epochs_dia_predicted = mne.EpochsArray(
        epoch_matrix_dia_predicted,
        info,
        tmin=tmin,
        baseline=baseline,
    )
    epochs_dia_predicted.set_montage(standard_1020_montage)
    epochs_dia_cleaned = mne.EpochsArray(
        epoch_matrix_dia_cleaned,
        info,
        tmin=tmin,
        baseline=baseline,
    )
    epochs_dia_cleaned.set_montage(standard_1020_montage)

    # Create mne evoked objects and append to lists
    evokeds_sys_before.append(epochs_sys_before.average())
    evokeds_sys_predicetd.append(epochs_sys_predicted.average())
    evokeds_sys_cleaned.append(epochs_sys_cleaned.average())
    evokeds_dia_before.append(epochs_dia_before.average())
    evokeds_dia_predicetd.append(epochs_dia_predicted.average())
    evokeds_dia_cleaned.append(epochs_dia_cleaned.average())

# Grand averages
ga_sys_before = mne.grand_average(evokeds_sys_before)
ga_sys_predicted = mne.grand_average(evokeds_sys_predicetd)
ga_sys_cleaned = mne.grand_average(evokeds_sys_cleaned)
ga_dia_before = mne.grand_average(evokeds_dia_before)
ga_dia_predicted = mne.grand_average(evokeds_dia_predicetd)
ga_dia_cleaned = mne.grand_average(evokeds_dia_cleaned)

# Plot stim-locked grand averages
xlim = [-0.7, 0.7]
ylim = [-5, 7]
topomap_times = [-0.23, 0, 0.17, 0.3]
ts_args = dict(gfp=True, xlim=xlim, ylim=dict(eeg=ylim))
topomap_args = dict(sensors=False, cmap=ccm, vmin=ylim[0], vmax=ylim[1])
ga_sys_before.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_specificity, "erp_sys_1_before.png"),
    dpi=300,
    transparent=True,
)
ga_sys_predicted.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_specificity, "erp_sys_2_predicted.png"),
    dpi=300,
    transparent=True,
)
ga_sys_cleaned.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_specificity, "erp_sys_3_cleaned.png"),
    dpi=300,
    transparent=True,
)
topomap_times = [-0.53, 0, 0.17, 0.3]
ga_dia_before.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_specificity, "erp_dia_1_before.png"),
    dpi=300,
    transparent=True,
)
ga_dia_predicted.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_specificity, "erp_dia_2_predicted.png"),
    dpi=300,
    transparent=True,
)
ga_dia_cleaned.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_specificity, "erp_dia_3_cleaned.png"),
    dpi=300,
    transparent=True,
)

# Define adjacency matrix
adjacency, channel_names = mne.channels.find_ch_adjacency(
    epochs_sys_before.info, ch_type="eeg"
)

# Re-organize data
X_sys_before = np.zeros((len(subject_list), len(erp_times_stimlock), len(channel_list)))
X_sys_cleaned = np.zeros(
    (len(subject_list), len(erp_times_stimlock), len(channel_list))
)
for s, subject in enumerate(subject_list):
    X_sys_before[s, :, :] = evokeds_sys_before[s].data.T
    X_sys_cleaned[s, :, :] = evokeds_sys_cleaned[s].data.T
X_dia_before = np.zeros((len(subject_list), len(erp_times_stimlock), len(channel_list)))
X_dia_cleaned = np.zeros(
    (len(subject_list), len(erp_times_stimlock), len(channel_list))
)
for s, subject in enumerate(subject_list):
    X_dia_before[s, :, :] = evokeds_dia_before[s].data.T
    X_dia_cleaned[s, :, :] = evokeds_dia_cleaned[s].data.T

# For an F test we need the degrees of freedom for the numerator
# (number of conditions - 1) and the denominator (number of observations
# - number of conditions):
n_conditions = 2
n_observations = len(subject_list)
df_effect = n_conditions - 1
df_error = n_observations - n_conditions

# Test before versus cleaned in sys trials =================================================
data_1 = X_sys_before
data_2 = X_sys_cleaned
evoked_1 = ga_sys_before
evoked_2 = ga_sys_cleaned
label_1 = "before"
label_2 = "cleaned"

topocmap = ccm
sigfill_color = "grey"
label_1_color = "red"
label_2_color = "darkcyan"

# Organize data
evokeds = {label_1: evoked_1, label_2: evoked_2}

# Calculate cluster-permutation statistics in channel-time space
cluster_stats = mne.stats.spatio_temporal_cluster_test(
    [data_1, data_2],
    n_permutations=1000,
    tail=1,
    n_jobs=-1,
    adjacency=adjacency,
)

# Unpack
F_obs, clusters, p_values, _ = cluster_stats

# Calculate effect sizes from F-stats
petasq_sys = np.divide((F_obs * df_effect), (F_obs * df_effect) + df_error)
adjpetasq_sys = petasq_sys - ((1 - petasq_sys) * (df_effect / df_error))

# Significant clusters
p_accept = 0.01
significant_cluster_idx = np.where(p_values < p_accept)[0]

# Define colors and linestyles
colors = {label_1: label_1_color, label_2: label_2_color}
linestyles = {label_1: "-", label_2: "-"}

# Loop clusters
for i_clu, clu_idx in enumerate(significant_cluster_idx):

    # Unpack cluster info
    time_idx, space_idx = np.squeeze(clusters[clu_idx])
    ch_idx = np.unique(space_idx)
    time_idx = np.unique(time_idx)

    # Get topogrphy for F-stat
    f_map = F_obs[time_idx, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = evoked_1.times[time_idx]
    cluster_limits_sys = sig_times[[0, -1]]

    # Create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_idx, :] = True

    # Initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 10))

    # Plot average test-statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], evoked_1.info, tmin=0)
    f_evoked.plot_topomap(
        times=0,
        time_format="",
        mask=mask,
        axes=ax_topo,
        cmap=topocmap,
        vmin=np.min,
        vmax=np.max,
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
    )
    image = ax_topo.images[0]

    # Create additional axes for ERF and colorbar
    divider = make_axes_locatable(ax_topo)

    # Add axes for colorbar
    ax_colorbar = divider.append_axes("right", size="8%", pad=0.5)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel("F-values ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]]))

    # Add new axis for time courses and plot time courses
    ax_signals = divider.append_axes("right", size="200%", pad=1.5)
    title = f"{len(ch_idx)} sensor"
    if len(ch_idx) > 1:
        title += "s"
    mne.viz.plot_compare_evokeds(
        evokeds,
        title=title,
        picks=ch_idx,
        axes=ax_signals,
        combine="mean",
        colors=colors,
        linestyles=linestyles,
        show=False,
        legend="upper right",
        split_legend=True,
        truncate_yaxis="auto",
        truncate_xaxis=False,
    )

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ymin, ymax = ymin * 1.1, ymax * 1.1
    ax_signals.fill_betweenx(
        (ymin, ymax), sig_times[0], sig_times[-1], color=sigfill_color, alpha=0.3
    )

    # Clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=0.05)
    plt.show()

    fig.savefig(
        os.path.join(
            path_results_specificity, f"specificity_sys_cluster_{i_clu+1}.png"
        ),
        dpi=300,
        transparent=True,
    )

# Test before versus cleaned in dia trials =================================================
data_1 = X_dia_before
data_2 = X_dia_cleaned
evoked_1 = ga_dia_before
evoked_2 = ga_dia_cleaned
label_1 = "before"
label_2 = "cleaned"

topocmap = ccm
sigfill_color = "grey"
label_1_color = "red"
label_2_color = "darkcyan"

# Organize data
evokeds = {label_1: evoked_1, label_2: evoked_2}

# Calculate cluster-permutation statistics in channel-time space
cluster_stats = mne.stats.spatio_temporal_cluster_test(
    [data_1, data_2],
    n_permutations=1000,
    tail=1,
    n_jobs=-1,
    adjacency=adjacency,
)

# Unpack
F_obs, clusters, p_values, _ = cluster_stats

# Calculate effect sizes from F-stats
petasq_dia = np.divide((F_obs * df_effect), (F_obs * df_effect) + df_error)
adjpetasq_dia = petasq_dia - ((1 - petasq_dia) * (df_effect / df_error))

# Significant clusters
p_accept = 0.01
significant_cluster_idx = np.where(p_values < p_accept)[0]

# Define colors and linestyles
colors = {label_1: label_1_color, label_2: label_2_color}
linestyles = {label_1: "-", label_2: "-"}

# Loop clusters
for i_clu, clu_idx in enumerate(significant_cluster_idx):

    # Unpack cluster info
    time_idx, space_idx = np.squeeze(clusters[clu_idx])
    ch_idx = np.unique(space_idx)
    time_idx = np.unique(time_idx)

    # Get topogrphy for F-stat
    f_map = F_obs[time_idx, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = evoked_1.times[time_idx]
    cluster_limits_dia = sig_times[[0, -1]]

    # Create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_idx, :] = True

    # Initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 10))

    # Plot average test-statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], evoked_1.info, tmin=0)
    f_evoked.plot_topomap(
        times=0,
        time_format="",
        mask=mask,
        axes=ax_topo,
        cmap=topocmap,
        vmin=np.min,
        vmax=np.max,
        show=False,
        colorbar=False,
        mask_params=dict(markersize=10),
    )
    image = ax_topo.images[0]

    # Create additional axes for ERF and colorbar
    divider = make_axes_locatable(ax_topo)

    # Add axes for colorbar
    ax_colorbar = divider.append_axes("right", size="8%", pad=0.5)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel("F-values ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]]))

    # Add new axis for time courses and plot time courses
    ax_signals = divider.append_axes("right", size="200%", pad=1.5)
    title = f"{len(ch_idx)} sensor"
    if len(ch_idx) > 1:
        title += "s"
    mne.viz.plot_compare_evokeds(
        evokeds,
        title=title,
        picks=ch_idx,
        axes=ax_signals,
        combine="mean",
        colors=colors,
        linestyles=linestyles,
        show=False,
        legend="upper right",
        split_legend=True,
        truncate_yaxis="auto",
        truncate_xaxis=False,
    )

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ymin, ymax = ymin * 1.1, ymax * 1.1
    ax_signals.fill_betweenx(
        (ymin, ymax), sig_times[0], sig_times[-1], color=sigfill_color, alpha=0.3
    )

    # Clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=0.05)
    plt.show()

    fig.savefig(
        os.path.join(
            path_results_specificity, f"specificity_dia_cluster_{i_clu+1}.png"
        ),
        dpi=300,
        transparent=True,
    )
     
# Plot stim-locked grand averages
xlim = [-0.7, 0.7]
ylim = [-0.15, 0.9]


ts_args = dict(gfp=False, xlim=xlim, ylim=dict(eeg=ylim), units=dict(eeg='η²', grad='fT/cm', mag='fT'))
topomap_args = dict(sensors=False, cmap="Oranges", vmin=ylim[0], vmax=ylim[1])

# Create and save apes sys plot
topomap_times = [-0.6, -0.3, -0.23, -0.15, 0, 0.2]
apes_sys = ga_sys_before
apes_sys.data = adjpetasq_sys.T / 1000000
apes_sys.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_specificity, "apes_erp_sys.png"),
    dpi=300,
    transparent=True,
)

# Create and save apes dia plot
topomap_times = [-0.53, -0.45, -0.3, -0.15, 0, 0.4]
apes_dia = ga_dia_before
apes_dia.data = adjpetasq_dia.T / 1000000
apes_dia.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_specificity, "apes_erp_dia.png"),
    dpi=300,
    transparent=True,
)
    
