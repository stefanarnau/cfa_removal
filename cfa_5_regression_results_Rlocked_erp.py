#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------------------
# This script plots the results from the CFA-removal in terms of ERPs for R-locked epochs
# (epochs without experimental stimulation).
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
path_predicted_data = "/mnt/data_heap/ecg_removal/6_predicted_data/"
path_file = "/home/plkn/repos/cfa_removal/"
path_meta = "/mnt/data_heap/ecg_removal/0_meta/"
path_results_efficacy = "/mnt/data_heap/ecg_removal/7_results/"

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
erp_times_rlock = (
    loadmat(os.path.join(path_meta, "time_rlock.mat"))["time_rlock"]
    .astype("float64")
    .ravel()
)

# Create a basic mne info structure
sfreq = 1000
info = mne.create_info(channel_names, sfreq, ch_types="eeg", verbose=None)

# Create a montage
standard_1020_montage = mne.channels.make_standard_montage("standard_1020")

# Lists for eveoked objects
evokeds_rlock_before = []
evokeds_rlock_predicetd = []
evokeds_rlock_after = []

# Iterate subjects nad create a (n_epochs, n_channels, n_times) array
for s, subject in enumerate(subject_list):

    print(f"\nProcessing {subject}...\n")

    # Get n-epochs r-lock
    dat = joblib.load(
        os.path.join(
            path_predicted_data,
            f"r_locked_y_before_{subject}_channel_{channel_list[0]}.joblib",
        ),
    )
    n_epochs_rlock = dat.shape[0]

    # Init subject epoch matrices
    epoch_matrix_rlock_before = np.zeros(
        (n_epochs_rlock, len(channel_list), len(erp_times_rlock))
    )
    epoch_matrix_rlock_predicted = np.zeros(
        (n_epochs_rlock, len(channel_list), len(erp_times_rlock))
    )
    epoch_matrix_rlock_after = np.zeros(
        (n_epochs_rlock, len(channel_list), len(erp_times_rlock))
    )

    # iterate channels
    for c, channel in enumerate(channel_list):

        # Load r-lock before
        dat = joblib.load(
            os.path.join(
                path_predicted_data,
                f"r_locked_y_before_{subject}_channel_{str(channel)}.joblib",
            ),
        )
        epoch_matrix_rlock_before[:, c, :] = dat * 1e-6

        # Load r-lock predicted
        dat = joblib.load(
            os.path.join(
                path_predicted_data,
                f"r_locked_y_predicted_{subject}_channel_{str(channel)}.joblib",
            ),
        )
        epoch_matrix_rlock_predicted[:, c, :] = dat * 1e-6

    # Substract predictions
    epoch_matrix_rlock_after = epoch_matrix_rlock_before - epoch_matrix_rlock_predicted

    # Create an mne epochs objects
    baseline = None
    tmin = erp_times_rlock[0] / sfreq
    epochs_rlock_before = mne.EpochsArray(
        epoch_matrix_rlock_before, info, tmin=tmin, baseline=baseline,
    )
    epochs_rlock_before.set_montage(standard_1020_montage)
    epochs_rlock_predicted = mne.EpochsArray(
        epoch_matrix_rlock_predicted, info, tmin=tmin, baseline=baseline,
    )
    epochs_rlock_predicted.set_montage(standard_1020_montage)
    epochs_rlock_after = mne.EpochsArray(
        epoch_matrix_rlock_after, info, tmin=tmin, baseline=baseline,
    )
    epochs_rlock_after.set_montage(standard_1020_montage)

    # Create an mne evoked objects and append to lists
    evokeds_rlock_before.append(epochs_rlock_before.average())
    evokeds_rlock_predicetd.append(epochs_rlock_predicted.average())
    evokeds_rlock_after.append(epochs_rlock_after.average())

# Grand averages
ga_rlock_before = mne.grand_average(evokeds_rlock_before)
ga_rlock_predicted = mne.grand_average(evokeds_rlock_predicetd)
ga_rlock_after = mne.grand_average(evokeds_rlock_after)

# Choose channels 
channel_list = ["FCz", "PO7", "PO8"]

# Get indices
channel_idx = [i for i, x in enumerate(channel_names) if x in channel_list]

# Loop datasets
ts_before = []
for evoked in evokeds_rlock_before:
    ts_before.append(evoked.data[channel_idx, :].mean(axis=0) * 1e6)
    
ts_after = [] 
for evoked in evokeds_rlock_after:
    ts_after.append(evoked.data[channel_idx, :].mean(axis=0) * 1e6)
    
# Stack
ts_before = np.stack(ts_before)
ts_after = np.stack(ts_after)
    
# Save
ts_stats_before = np.vstack((ts_before.mean(axis=0), ts_before.std(axis=0))).T
ts_stats_after = np.vstack((ts_after.mean(axis=0), ts_after.std(axis=0))).T
ts_times = ga_rlock_after.times
np.savetxt(os.path.join(path_results_efficacy, "residual_times.csv"), ts_times, delimiter="\t")
np.savetxt(os.path.join(path_results_efficacy, "interindividual_variance_before.csv"), ts_stats_before, delimiter="\t")
np.savetxt(os.path.join(path_results_efficacy, "interindividual_variance_after.csv"), ts_stats_after, delimiter="\t")

# Plot r-locked grand averages
xlim = [-0.2, 0.4]
ylim = [-5, 7]
topomap_times = [-0.02, 0.01, 0.25]
ts_args = dict(gfp=True, xlim=xlim, ylim=dict(eeg=ylim))
topomap_args = dict(sensors=False, cmap=ccm, vlim=(ylim[0], ylim[1]))

ga_rlock_before.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_efficacy, "erp_rlocked_1_before.png"),
    dpi=300,
    transparent=True,
)
ga_rlock_predicted.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_efficacy, "erp_rlocked_2_predicted.png"),
    dpi=300,
    transparent=True,
)
ga_rlock_after.plot_joint(
    title=None,
    ts_args=ts_args,
    topomap_args=topomap_args,
    times=topomap_times,
    show=False,
)
plt.savefig(
    os.path.join(path_results_efficacy, "erp_rlocked_3_cleaned.png"),
    dpi=300,
    transparent=True,
)


aa=bb
# Define adjacency matrix
adjacency, channel_names = mne.channels.find_ch_adjacency(
    epochs_rlock_before.info, ch_type="eeg"
)

# Plot adjacency matrix
# plt.imshow(adjacency.toarray(), cmap=ccm, origin="lower", interpolation="nearest")
# plt.xlabel(f"{len(channel_names)} electrodes")
# plt.ylabel(f"{len(channel_names)} electrodes")
# plt.title("electrode adjacency")

# Re-organize data
X_before = np.zeros((len(subject_list), len(erp_times_rlock), len(channel_list)))
X_cleaned = np.zeros((len(subject_list), len(erp_times_rlock), len(channel_list)))
for s, subject in enumerate(subject_list):
    X_before[s, :, :] = evokeds_rlock_before[s].data.T
    X_cleaned[s, :, :] = evokeds_rlock_after[s].data.T

# Test before versus cleaned =================================================
data_1 = X_before
data_2 = X_cleaned
evoked_1 = ga_rlock_before
evoked_2 = ga_rlock_after
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
    [data_1, data_2], n_permutations=1000, tail=1, n_jobs=-1, adjacency=adjacency,
)

# Unpack
T_obs, clusters, p_values, _ = cluster_stats

# Significant clusters
p_accept = 0.01
significant_cluster_idx = np.where(p_values < p_accept)[0]

# Define colors and linestyles
colors = {label_1: label_1_color, label_2: label_2_color}
linestyles = {label_1: "-", label_2: "-"}

# Loop clusters
clust_times = []
for i_clu, clu_idx in enumerate(significant_cluster_idx):

    # Unpack cluster info
    time_idx, space_idx = np.squeeze(clusters[clu_idx])
    ch_idx = np.unique(space_idx)
    time_idx = np.unique(time_idx)

    clust_times.append(
        {
            "time": (
                evoked_1.times[time_idx].min() * 1000,
                evoked_1.times[time_idx].max() * 1000,
            )
        }
    )

    # Get topogrphy for F-stat
    f_map = T_obs[time_idx, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = evoked_1.times[time_idx]

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
        os.path.join(path_results_efficacy, f"efficacy_cluster_{i_clu+1}.png"),
        dpi=300,
        transparent=True,
    )

